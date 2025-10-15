//! Memory limits for async blocking tasks

use libc::RLIMIT_AS;
use std::{future::Future, io};
use tokio::sync::oneshot;

/// Run future in a forked child with a memory limit, report the result via `memory_failure_tx`.
/// Must be called from a blocking context (`tokio::task::spawn_blocking`).
pub fn spawn_blocking_with_memory_limit<F: Future<Output = ()> + Send + 'static>(
    memory_limit_bytes: u64,
    memory_failure_tx: oneshot::Sender<Result<(), std::io::Error>>,
    fut: F,
) {
    // Do NOT enter a Tokio runtime here. We are on a blocking thread
    let status = run_with_memory_limit(memory_limit_bytes, || {
        // this is run in the child - build a new single-thread runtime and drive the future
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(fut);
    });

    let res = status.and_then(|st| {
        if libc::WIFEXITED(st) {
            let code = libc::WEXITSTATUS(st);
            if code == 0 {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("child exited with code {code}"),
                ))
            }
        } else if libc::WIFSIGNALED(st) {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("child signaled: {}", libc::WTERMSIG(st)),
            ))
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::Other, format!("bad exit: {st}")))
        }
    });

    let _ = memory_failure_tx.send(res);
}

#[allow(unreachable_code)]
fn run_with_memory_limit<F>(memory_limit_bytes: u64, f: F) -> io::Result<i32>
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(move || {
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            return Err(io::Error::last_os_error());
        }
        if pid == 0 {
            let lim = libc::rlimit { rlim_cur: memory_limit_bytes, rlim_max: memory_limit_bytes };
            if unsafe { libc::setrlimit(RLIMIT_AS, &lim) } != 0 {
                unsafe { libc::_exit(251) };
            }
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f()))
                .map(|_| {
                    // success path
                    unsafe { libc::_exit(0) };
                })
                .map_err(|_| {
                    unsafe { libc::_exit(101) };
                });
        }
        let mut status = 0;
        if unsafe { libc::waitpid(pid, &mut status, 0) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(status)
    })
    .join()
    .unwrap()
}

// #[cfg(all(test, target_os = "linux"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TaskManager, TaskSpawner};
    use std::future::Future;
    use tokio::{
        sync::oneshot,
        time::{timeout, Duration},
    };

    // Helper: run the API and collect the oneshot result with a timeout.
    async fn run_case<F>(
        limit: u64,
        fut: F,
    ) -> Result<Result<(), io::Error>, tokio::time::error::Elapsed>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        tokio::task::spawn_blocking(move || {
            spawn_blocking_with_memory_limit(limit, tx, fut);
        });

        // Give the child up to 15s to finish (slow CI boxes, debug mode).
        timeout(Duration::from_secs(15), async move { rx.await.expect("sender dropped") }).await
    }

    /// A future that allocates ~8 MiB and exits.
    fn small_alloc_future() -> impl Future<Output = ()> + Send + 'static {
        async move {
            let size = 8usize * 1024 * 1024;
            let mut v = Vec::<u8>::with_capacity(size);
            // force commit to be conservative
            v.resize(size, 42u8);
            // a brief yield to exercise the timer; ensures runtime is alive
            tokio::time::sleep(Duration::from_millis(10)).await;
            drop(v);
        }
    }

    /// A future that tries to allocate 1 GiB. With a 256 MiB cap this should fail.
    fn huge_alloc_future() -> impl Future<Output = ()> + Send + 'static {
        async move {
            // This should fail at allocation time under RLIMIT_AS.
            // 1 GiB
            let size = 1024usize * 1024 * 1024;
            let mut v = Vec::<u8>::with_capacity(size);
            // If it somehow succeeded, set_len would make it "logically" that big.
            unsafe { v.set_len(size) };
            // Keep it around briefly to avoid instant free.
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(v);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn under_limit_succeeds() {
        let limit = 256u64 * 1024 * 1024; // 256 MiB
        let res = run_case(limit, small_alloc_future()).await.expect("timed out");
        assert!(res.is_ok(), "expected Ok(()), got {res:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn exceeding_limit_returns_error() {
        let limit = 256u64 * 1024 * 1024; // 256 MiB
        let res = run_case(limit, huge_alloc_future()).await.expect("timed out");
        assert!(res.is_err(), "expected Err, got Ok(())");

        let msg = res.err().unwrap().to_string();
        // Typical kernel/allocator outcomes when exceeding RLIMIT_AS:
        //  - SIGABRT (allocator abort on OOM)
        //  - SIGSEGV (bad access)
        //  - SIGKILL (rare here, but possible)
        // (We also accept an explicit non-zero exit code if the runtime bubbles it out.)
        assert!(
            msg.contains("child signaled:") || msg.contains("child exited with code"),
            "unexpected error message: {msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn panic_in_child_reports_nonzero_exit() {
        let limit = 512u64 * 1024 * 1024; // ample memory so panic path isn't masked
        let fut = async move {
            panic!("intentional panic in child");
        };

        let res = run_case(limit, fut).await.expect("timed out");
        assert!(res.is_err());

        let msg = res.err().unwrap().to_string();
        // Rust panics in the main task typically produce exit code 101.
        assert!(
            msg.contains("child exited with code") || msg.contains("bad exit"),
            "expected non-zero exit code, got: {msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_runs_isolated() {
        let limit = 256u64 * 1024 * 1024;

        let (tx1, rx1) = oneshot::channel();
        let (tx2, rx2) = oneshot::channel();

        let fut_ok = small_alloc_future();
        let fut_oom = huge_alloc_future();

        // Kick off both in parallel on blocking threads.
        let h1 = tokio::task::spawn_blocking(move || {
            spawn_blocking_with_memory_limit(limit, tx1, fut_ok);
        });
        let h2 = tokio::task::spawn_blocking(move || {
            spawn_blocking_with_memory_limit(limit, tx2, fut_oom);
        });

        let r1 = timeout(Duration::from_secs(15), rx1)
            .await
            .expect("timeout r1")
            .expect("sender dropped r1");
        let r2 = timeout(Duration::from_secs(15), rx2)
            .await
            .expect("timeout r2")
            .expect("sender dropped r2");

        // Ensure joins don't panic.
        let _ = (h1.await, h2.await);

        assert!(r1.is_ok(), "first should succeed: {r1:?}");
        assert!(r2.is_err(), "second should fail: {r2:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn executor_spawn_blocking_with_memory_limit_ok_and_oom() {
        // Use the current test runtime; do NOT create a nested runtime here.
        let handle = tokio::runtime::Handle::current().clone();
        let manager = TaskManager::new(handle.clone());
        let executor = manager.executor();

        // ------- OK future --------
        let (ok_tx, ok_rx) = oneshot::channel();
        let ok_fut = Box::pin(async move {
            let size = 8usize * 1024 * 1024; // 8 MiB
            let mut v = Vec::<u8>::with_capacity(size);
            v.resize(size, 7);
            tokio::time::sleep(Duration::from_millis(10)).await;
            drop(v);
        });
        let ok_jh = executor.spawn_blocking_with_memory_limit(256 * 1024 * 1024, ok_tx, ok_fut);

        // ------- OOM future --------
        let (oom_tx, oom_rx) = oneshot::channel();
        let oom_fut = Box::pin(async move {
            let size = 1024usize * 1024 * 1024; // 1 GiB
            let mut v = Vec::<u8>::with_capacity(size);
            unsafe { v.set_len(size) };
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(v);
        });
        let oom_jh = executor.spawn_blocking_with_memory_limit(256 * 1024 * 1024, oom_tx, oom_fut);

        // Await results with timeouts.
        let ok_res = timeout(Duration::from_secs(15), ok_rx)
            .await
            .expect("timeout ok")
            .expect("sender dropped ok");
        let oom_res = timeout(Duration::from_secs(15), oom_rx)
            .await
            .expect("timeout oom")
            .expect("sender dropped oom");

        // Join spawned tasks.
        let _ = (ok_jh.await, oom_jh.await);

        assert!(ok_res.is_ok(), "expected Ok(()), got {ok_res:?}");
        assert!(oom_res.is_err(), "expected Err for OOM, got Ok(())");
        let msg = oom_res.err().unwrap().to_string();
        assert!(
            msg.contains("child signaled:") || msg.contains("child exited with code"),
            "unexpected error message: {msg}"
        );
    }
}
