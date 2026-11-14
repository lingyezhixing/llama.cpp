#include "server-asr-job.h"

#include "server-common.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

// ETA uses the throughput of the most recent completed chunks
static constexpr size_t ETA_WINDOW = 5;
// the worker idles in short waits so a pause or a cancel is noticed quickly
static constexpr int    IDLE_WAIT_MS = 100;

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool is_terminal(server_asr_job_status status) {
    return status == server_asr_job_status::SUCCEEDED ||
           status == server_asr_job_status::FAILED    ||
           status == server_asr_job_status::CANCELLED;
}

struct server_asr_job_manager::impl {
    server_asr_offline_runner * runner = nullptr;
    size_t  queue_capacity = 8;
    int64_t retention_ms   = 24 * 60 * 60 * 1000;

    struct job {
        std::string           id;
        server_asr_job_input  input;
        server_asr_job_status status = server_asr_job_status::QUEUED;
        std::string           paused_reason;

        server_asr_timing timing;
        int64_t           audio_processed_ms = 0;
        int64_t           audio_total_ms     = 0;

        server_asr_usage usage;
        bool             has_usage = false;

        server_asr_result result;
        bool              has_result = false;

        server_asr_error error;

        server_asr_run_state state;

        std::atomic<bool> cancel_requested{false};

        // monotonic bookkeeping
        int64_t created_ms        = 0;
        int64_t finished_ms       = 0;
        int64_t processing_ms     = 0; // accumulated, excludes queue and pause time
        int64_t processing_start  = 0; // 0 when not running
        int64_t last_chunk_ms     = 0; // processing time at the last progress callback
        int64_t last_chunk_audio  = 0;
        std::vector<std::pair<int64_t, int64_t>> recent; // (audio_ms delta, processing_ms delta)
    };

    mutable std::mutex      mutex;
    std::condition_variable cv;

    std::map<std::string, std::unique_ptr<job>> jobs;
    std::deque<std::string> queue;
    std::string current; // running or paused job id

    std::atomic<bool> realtime{false};
    bool terminating = false;

    std::thread worker;

    // ---- helpers, all called with the mutex held ----

    int64_t active_processing_ms(const job & j, int64_t now) const {
        return j.processing_ms + (j.processing_start > 0 ? now - j.processing_start : 0);
    }

    void enter_running(job & j, int64_t now) {
        if (j.timing.started_at.empty()) {
            j.timing.started_at = server_asr_now_rfc3339();
        }
        j.status          = server_asr_job_status::RUNNING;
        j.processing_start = now;
        j.last_chunk_ms    = active_processing_ms(j, now);
        j.last_chunk_audio = j.audio_processed_ms;
    }

    void leave_running(job & j, int64_t now) {
        if (j.processing_start > 0) {
            j.processing_ms  += now - j.processing_start;
            j.processing_start = 0;
        }
        j.timing.processing_ms = j.processing_ms;
    }

    void enter_paused(job & j, int64_t now) {
        leave_running(j, now);
        j.status        = server_asr_job_status::PAUSED;
        j.paused_reason = "realtime_priority";
        j.timing.paused_at = server_asr_now_rfc3339();
    }

    void finish(job & j, server_asr_job_status status, int64_t now) {
        if (j.processing_start > 0) {
            leave_running(j, now);
        }
        j.status           = status;
        j.finished_ms      = now;
        j.timing.finished_at = server_asr_now_rfc3339();
        j.timing.processing_ms = j.processing_ms;

        // the audio file is only useful while the job can still run
        if (!j.input.audio_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(j.input.audio_path, ec);
            j.input.audio_path.clear();
        }
    }

    void update_eta(job & j, int64_t now) {
        j.timing.eta_ms = 0;
        j.timing.estimated_end_at.clear();

        if (j.status != server_asr_job_status::RUNNING || j.audio_total_ms <= 0 || j.recent.empty()) {
            return;
        }

        int64_t recent_audio = 0;
        int64_t recent_proc  = 0;
        for (const auto & [audio, proc] : j.recent) {
            recent_audio += audio;
            recent_proc  += proc;
        }
        if (recent_audio <= 0 || recent_proc <= 0) {
            return;
        }

        const int64_t remaining = std::max<int64_t>(0, j.audio_total_ms - j.audio_processed_ms);
        const int64_t eta       = remaining * recent_proc / recent_audio;

        j.timing.eta_ms = eta;
        j.timing.estimated_end_at = server_asr_rfc3339_after_ms(eta);
    }

    void expire_finished(int64_t now) {
        for (auto it = jobs.begin(); it != jobs.end();) {
            auto & j = *it->second;
            if (is_terminal(j.status) && j.finished_ms > 0 && now - j.finished_ms > retention_ms) {
                if (!j.input.audio_path.empty()) {
                    std::error_code ec;
                    std::filesystem::remove(j.input.audio_path, ec);
                }
                it = jobs.erase(it);
            } else {
                ++it;
            }
        }
    }

    server_asr_job_view view_locked(const job & j) const {
        server_asr_job_view out;

        out.id            = j.id;
        out.status        = j.status;
        out.paused_reason = j.paused_reason;
        out.timing        = j.timing;
        out.audio_processed_ms = j.audio_processed_ms;
        out.audio_total_ms     = j.audio_total_ms;
        out.has_usage     = j.has_usage;
        out.usage         = j.usage;
        out.has_result    = j.has_result;
        out.include_items = j.input.timestamps;
        out.result        = j.result;
        out.error         = j.error;

        return out;
    }

    // ---- worker ----

    void run_step();

    void worker_loop() {
        std::unique_lock<std::mutex> lock(mutex);

        while (!terminating) {
            const int64_t now = now_ms();
            expire_finished(now);

            if (!current.empty()) {
                auto it = jobs.find(current);
                if (it == jobs.end()) {
                    current.clear();
                    continue;
                }

                job & j = *it->second;

                if (j.status == server_asr_job_status::PAUSED) {
                    if (realtime.load()) {
                        cv.wait_for(lock, std::chrono::milliseconds(IDLE_WAIT_MS));
                        continue;
                    }
                    // realtime ended: resume before any queued job
                    enter_running(j, now);
                    j.timing.paused_at.clear();
                    j.paused_reason.clear();
                }

                lock.unlock();
                run_step();
                lock.lock();
                continue;
            }

            if (queue.empty() || realtime.load()) {
                cv.wait_for(lock, std::chrono::milliseconds(IDLE_WAIT_MS));
                continue;
            }

            current = queue.front();
            queue.pop_front();

            auto it = jobs.find(current);
            if (it == jobs.end()) {
                current.clear();
                continue;
            }

            enter_running(*it->second, now);

            lock.unlock();
            run_step();
            lock.lock();
        }
    }
};

void server_asr_job_manager::impl::run_step() {
    server_asr_job_input input;
    server_asr_run_state state;
    std::atomic<bool> * cancel = nullptr;

    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = jobs.find(current);
        if (it == jobs.end()) {
            return;
        }
        job & j = *it->second;
        // a cancelled job still gets one more step so the runner can stop at a
        // safe point and report CANCELLED
        if (j.status != server_asr_job_status::RUNNING && j.status != server_asr_job_status::CANCELLING) {
            return;
        }
        input  = j.input;
        state  = j.state;
        cancel = &j.cancel_requested;
    }

    server_asr_error err;

    const auto pause_requested = [this]() { return realtime.load(); };
    const auto progress = [this](int64_t audio_ms, const server_asr_usage & usage) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = jobs.find(current);
        if (it == jobs.end()) {
            return;
        }
        job & j = *it->second;

        const int64_t now = now_ms();
        const int64_t proc_now = active_processing_ms(j, now);

        const int64_t audio_delta = audio_ms - j.last_chunk_audio;
        const int64_t proc_delta  = proc_now - j.last_chunk_ms;
        if (audio_delta > 0 && proc_delta > 0) {
            j.recent.emplace_back(audio_delta, proc_delta);
            while (j.recent.size() > ETA_WINDOW) {
                j.recent.erase(j.recent.begin());
            }
            j.last_chunk_audio = audio_ms;
            j.last_chunk_ms    = proc_now;
        }

        j.audio_processed_ms = audio_ms;
        j.usage              = usage;
        j.has_usage          = true;

        update_eta(j, now);
    };

    const server_asr_run_status status = runner->run(input, state, *cancel, pause_requested, progress, err);

    std::lock_guard<std::mutex> lock(mutex);
    auto it = jobs.find(current);
    if (it == jobs.end()) {
        return;
    }
    job & j = *it->second;

    const int64_t now = now_ms();

    j.state = state;

    switch (status) {
        case server_asr_run_status::DONE:
            j.result      = state.result;
            j.has_result  = true;
            j.usage       = state.usage;
            j.has_usage   = true;
            j.audio_processed_ms = j.audio_total_ms > 0 ? j.audio_total_ms : state.audio_processed_ms;
            j.timing.eta_ms = 0;
            j.timing.estimated_end_at.clear();
            finish(j, server_asr_job_status::SUCCEEDED, now);
            current.clear();
            cv.notify_all();
            break;

        case server_asr_run_status::PAUSED:
            enter_paused(j, now);
            // current stays set, the worker resumes it when realtime ends
            break;

        case server_asr_run_status::CANCELLED:
            j.audio_processed_ms = state.audio_processed_ms;
            finish(j, server_asr_job_status::CANCELLED, now);
            current.clear();
            cv.notify_all();
            break;

        case server_asr_run_status::FAILED:
            j.error = err;
            if (err.empty()) {
                j.error = { "internal_error", "the ASR worker failed", "" };
            }
            j.audio_processed_ms = state.audio_processed_ms;
            finish(j, server_asr_job_status::FAILED, now);
            current.clear();
            cv.notify_all();
            break;
    }
}

server_asr_job_manager::server_asr_job_manager(server_asr_offline_runner * runner, size_t queue_capacity, int64_t retention_ms)
    : pimpl(std::make_unique<impl>()) {
    pimpl->runner         = runner;
    pimpl->queue_capacity = queue_capacity == 0 ? 1 : queue_capacity;
    pimpl->retention_ms   = retention_ms;

    pimpl->worker = std::thread([this] { pimpl->worker_loop(); });
}

server_asr_job_manager::~server_asr_job_manager() {
    terminate();
}

std::string server_asr_job_manager::submit(server_asr_job_input && input, server_asr_error & err) {
    err = {};

    if (pimpl->runner == nullptr) {
        err = { "asr_not_implemented", "no offline runner is available", "" };
        return "";
    }

    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);

        size_t queued = pimpl->queue.size();
        if (!pimpl->current.empty()) {
            queued++;
        }
        if (queued >= pimpl->queue_capacity) {
            err = { "queue_full", "the offline queue is full", "" };
            return "";
        }
    }

    int64_t total_ms = 0;
    if (!pimpl->runner->probe(input.audio_path, total_ms, err)) {
        if (err.empty()) {
            err = { "invalid_audio", "the audio cannot be decoded", "" };
        }
        return "";
    }

    static std::atomic<uint64_t> counter{0};
    const uint64_t n = counter.fetch_add(1);

    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "%08" PRIx64, n);

    const int64_t now = now_ms();

    auto job = std::make_unique<impl::job>();
    job->id                = std::string("asr_") + suffix;
    job->input             = std::move(input);
    job->audio_total_ms    = total_ms;
    job->created_ms        = now;
    job->timing.created_at = server_asr_now_rfc3339();
    job->status            = server_asr_job_status::QUEUED;

    const std::string id = job->id;

    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        pimpl->jobs[id] = std::move(job);
        pimpl->queue.push_back(id);
    }

    pimpl->cv.notify_all();

    return id;
}

bool server_asr_job_manager::get(const std::string & id, server_asr_job_view & out, server_asr_error & err) const {
    err = {};

    std::lock_guard<std::mutex> lock(pimpl->mutex);

    const auto it = pimpl->jobs.find(id);
    if (it == pimpl->jobs.end()) {
        err = { "job_not_found", "unknown transcription id", "id" };
        return false;
    }

    out = pimpl->view_locked(*it->second);
    return true;
}

server_asr_cancel_result server_asr_job_manager::cancel(const std::string & id, server_asr_error & err) {
    err = {};

    std::lock_guard<std::mutex> lock(pimpl->mutex);

    const auto it = pimpl->jobs.find(id);
    if (it == pimpl->jobs.end()) {
        err = { "job_not_found", "unknown transcription id", "id" };
        return server_asr_cancel_result::NOT_FOUND;
    }

    impl::job & j = *it->second;
    const int64_t now = now_ms();

    if (is_terminal(j.status)) {
        pimpl->jobs.erase(it);
        return server_asr_cancel_result::DELETED;
    }

    if (j.status == server_asr_job_status::RUNNING) {
        j.cancel_requested.store(true);
        j.status = server_asr_job_status::CANCELLING;
        return server_asr_cancel_result::CANCELLING;
    }

    // queued or paused: drop it right away
    if (j.status == server_asr_job_status::PAUSED && pimpl->current == id) {
        pimpl->current.clear();
    } else {
        const auto qi = std::find(pimpl->queue.begin(), pimpl->queue.end(), id);
        if (qi != pimpl->queue.end()) {
            pimpl->queue.erase(qi);
        }
    }

    pimpl->finish(j, server_asr_job_status::CANCELLED, now);
    pimpl->cv.notify_all();

    return server_asr_cancel_result::CANCELLED;
}

void server_asr_job_manager::set_realtime_active(bool value) {
    pimpl->realtime.store(value);
    pimpl->cv.notify_all();
}

bool server_asr_job_manager::realtime_active() const {
    return pimpl->realtime.load();
}

void server_asr_job_manager::terminate() {
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        if (pimpl->terminating) {
            return;
        }
        pimpl->terminating = true;

        for (auto & [id, job] : pimpl->jobs) {
            if (!is_terminal(job->status)) {
                job->cancel_requested.store(true);
            }
        }
    }

    pimpl->cv.notify_all();

    if (pimpl->worker.joinable()) {
        if (pimpl->worker.get_id() == std::this_thread::get_id()) {
            pimpl->worker.detach();
        } else {
            pimpl->worker.join();
        }
    }

    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (auto & [id, job] : pimpl->jobs) {
        if (!job->input.audio_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(job->input.audio_path, ec);
            job->input.audio_path.clear();
        }
    }
    pimpl->jobs.clear();
    pimpl->queue.clear();
    pimpl->current.clear();
}
