#pragma once

#include "SpectralTypesSoA.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace spectile::core {

// ============================================================================
// LockFreeTripleBuffer<T> — Wait-Free O(1) Single-Producer / Single-Consumer
// ============================================================================
/**
 * @brief Wait-free O(1) Triple Buffer for transferring large SoA structures
 *        (such as MaskMatrix2D or GenomeState) between threads without locks or tearing.
 *
 * State byte encoding:
 *   - Bits [1:0] (0x03): Index of the shared (middle) buffer in {0, 1, 2}.
 *   - Bit  [2]   (0x04): Dirty flag indicating a new frame was published by the producer.
 */
template <typename T>
class alignas(kCacheLineBytes) LockFreeTripleBuffer {
public:
    static constexpr std::uint8_t kIndexMask = 0x03u;
    static constexpr std::uint8_t kDirtyBit  = 0x04u;

    LockFreeTripleBuffer() {
        static_assert(std::atomic<std::uint8_t>::is_always_lock_free,
                      "std::atomic<uint8_t> must be lock-free");
    }

    explicit LockFreeTripleBuffer(const T& initialValue)
        : buffers_{initialValue, initialValue, initialValue} {}

    LockFreeTripleBuffer(const LockFreeTripleBuffer&)            = delete;
    LockFreeTripleBuffer& operator=(const LockFreeTripleBuffer&) = delete;

    /**
     * @brief Returns a mutable reference to the producer's private back-buffer.
     */
    [[nodiscard]] T& writeBuffer() noexcept {
        return buffers_[writeIndex_];
    }

    /**
     * @brief Publishes the current writeBuffer() in wait-free O(1) time.
     */
    void publish() noexcept {
        std::uint8_t expected = state_.load(std::memory_order_relaxed);
        std::uint8_t desired  = 0;
        do {
            desired = static_cast<std::uint8_t>((writeIndex_ & kIndexMask) | kDirtyBit);
        } while (!state_.compare_exchange_weak(
            expected, desired, std::memory_order_acq_rel, std::memory_order_relaxed));

        writeIndex_ = static_cast<std::uint8_t>(expected & kIndexMask);
    }

    /**
     * @brief Convenience method to copy a value into the write buffer and publish it.
     */
    void writeAndPublish(const T& value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        buffers_[writeIndex_] = value;
        publish();
    }

    /**
     * @brief Consumer-side wait-free acquisition of the latest published buffer.
     * @return true if a newly published buffer was swapped in, false if unchanged.
     */
    bool pullLatest() noexcept {
        std::uint8_t current = state_.load(std::memory_order_relaxed);
        if ((current & kDirtyBit) == 0) {
            return false;
        }

        std::uint8_t desired = 0;
        do {
            if ((current & kDirtyBit) == 0) {
                return false;
            }
            desired = static_cast<std::uint8_t>(readIndex_ & kIndexMask);
        } while (!state_.compare_exchange_weak(
            current, desired, std::memory_order_acq_rel, std::memory_order_relaxed));

        readIndex_ = static_cast<std::uint8_t>(current & kIndexMask);
        return true;
    }

    /**
     * @brief Returns a const reference to the consumer's current front-buffer.
     */
    [[nodiscard]] const T& readBuffer() const noexcept {
        return buffers_[readIndex_];
    }

    /**
     * @brief Returns a mutable reference to the consumer's current front-buffer
     *        (useful when the audio thread performs in-place normalization).
     */
    [[nodiscard]] T& mutableReadBuffer() noexcept {
        return buffers_[readIndex_];
    }

    /**
     * @brief Checks whether an unread update is pending without consuming it.
     */
    [[nodiscard]] bool hasNewData() const noexcept {
        return (state_.load(std::memory_order_acquire) & kDirtyBit) != 0;
    }

private:
    alignas(kCacheLineBytes) std::array<T, 3> buffers_{};
    alignas(kCacheLineBytes) std::atomic<std::uint8_t> state_{1u}; // Shared index starts at 1
    alignas(kCacheLineBytes) std::uint8_t writeIndex_{0u};         // Producer starts at 0
    alignas(kCacheLineBytes) std::uint8_t readIndex_{2u};          // Consumer starts at 2
};

// ============================================================================
// LockFreeSpscRingBuffer<T, Capacity> — Wait-Free Cache-Aligned SPSC Queue
// ============================================================================
/**
 * @brief Wait-free Single-Producer / Single-Consumer Ring Buffer with cache-line
 *        padding and shadow index caching to eliminate false sharing.
 *        Designed for streaming SpectralColumnPacket (6.65 KB/hop) from the
 *        Real-Time Audio Thread to the 60/120 FPS GPU UI Renderer.
 */
template <typename T, std::size_t Capacity = 64>
class alignas(kCacheLineBytes) LockFreeSpscRingBuffer {
public:
    static_assert(Capacity >= 2 && ((Capacity & (Capacity - 1)) == 0),
                  "LockFreeSpscRingBuffer Capacity must be a power of 2 >= 2");
    static constexpr std::size_t kMask = Capacity - 1;

    LockFreeSpscRingBuffer() = default;
    LockFreeSpscRingBuffer(const LockFreeSpscRingBuffer&)            = delete;
    LockFreeSpscRingBuffer& operator=(const LockFreeSpscRingBuffer&) = delete;

    /**
     * @brief Pushes an element from the Producer (Audio Thread) without blocking.
     * @return false if the ring buffer is full.
     */
    [[nodiscard]] bool tryPush(const T& item) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        const std::size_t writePos = writePos_.load(std::memory_order_relaxed);
        const std::size_t nextPos  = (writePos + 1) & kMask;

        if (nextPos == cachedReadPos_) {
            cachedReadPos_ = readPos_.load(std::memory_order_acquire);
            if (nextPos == cachedReadPos_) {
                return false; // Queue full
            }
        }

        storage_[writePos] = item;
        writePos_.store(nextPos, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pushes an element by move from the Producer without blocking.
     */
    [[nodiscard]] bool tryPushMove(T&& item) noexcept(std::is_nothrow_move_assignable_v<T>) {
        const std::size_t writePos = writePos_.load(std::memory_order_relaxed);
        const std::size_t nextPos  = (writePos + 1) & kMask;

        if (nextPos == cachedReadPos_) {
            cachedReadPos_ = readPos_.load(std::memory_order_acquire);
            if (nextPos == cachedReadPos_) {
                return false;
            }
        }

        storage_[writePos] = std::move(item);
        writePos_.store(nextPos, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pops the oldest available element on the Consumer (GPU/UI Thread).
     * @return false if the ring buffer is empty.
     */
    [[nodiscard]] bool tryPop(T& outItem) noexcept(std::is_nothrow_move_assignable_v<T>) {
        const std::size_t readPos = readPos_.load(std::memory_order_relaxed);

        if (readPos == cachedWritePos_) {
            cachedWritePos_ = writePos_.load(std::memory_order_acquire);
            if (readPos == cachedWritePos_) {
                return false; // Queue empty
            }
        }

        outItem = std::move(storage_[readPos]);
        readPos_.store((readPos + 1) & kMask, std::memory_order_release);
        return true;
    }

    /**
     * @brief Drains all queued items and keeps only the most recent one.
     * @return Number of items consumed.
     */
    std::size_t popLatest(T& outLatestItem) noexcept(std::is_nothrow_move_assignable_v<T>) {
        std::size_t poppedCount = 0;
        while (tryPop(outLatestItem)) {
            ++poppedCount;
        }
        return poppedCount;
    }

    [[nodiscard]] std::size_t sizeApprox() const noexcept {
        const std::size_t w = writePos_.load(std::memory_order_acquire);
        const std::size_t r = readPos_.load(std::memory_order_acquire);
        return (w - r) & kMask;
    }

    [[nodiscard]] bool empty() const noexcept {
        return sizeApprox() == 0;
    }

    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return Capacity - 1;
    }

    void clear() noexcept {
        const std::size_t w = writePos_.load(std::memory_order_acquire);
        readPos_.store(w, std::memory_order_release);
        cachedReadPos_  = w;
        cachedWritePos_ = w;
    }

private:
    alignas(kCacheLineBytes) std::atomic<std::size_t> writePos_{0};
    std::size_t                                       cachedReadPos_{0};

    alignas(kCacheLineBytes) std::atomic<std::size_t> readPos_{0};
    std::size_t                                       cachedWritePos_{0};

    alignas(kCacheLineBytes) std::array<T, Capacity>  storage_{};
};

// ============================================================================
// Real-Time RCU Pattern: DeferredSampleGarbageCollector & SpectralSampleSlot
// ============================================================================
/**
 * @brief Real-Time Safe Deferred Garbage Collector (Section 3.2.1).
 *
 * Guarantee:
 *   When a sample is replaced or unloaded in a SpectralSampleSlot, its shared_ptr
 *   is retained inside the collector's retirement pool. As long as the collector
 *   holds a reference, the reference count on the Audio Thread is >= 2.
 *   Therefore, when the Audio Thread releases its reference at the end of processBlock(),
 *   the ref-count drops from 2 -> 1 (NEVER 1 -> 0).
 *   A 10 Hz timer on the MessageThread periodically calls collectGarbage(), which
 *   safely destroys retired samples whose use_count() == 1 off the real-time thread.
 */
class DeferredSampleGarbageCollector {
public:
    using SamplePtr = std::shared_ptr<const SpectralSampleData>;

    explicit DeferredSampleGarbageCollector(std::size_t initialReserve = 64) {
        retiredPool_.reserve(initialReserve);
    }

    DeferredSampleGarbageCollector(const DeferredSampleGarbageCollector&)            = delete;
    DeferredSampleGarbageCollector& operator=(const DeferredSampleGarbageCollector&) = delete;

    /**
     * @brief Retires an old sample pointer from a non-audio thread (Worker or Message Thread)
     *        or enqueues it via the lock-free SPSC retirement queue.
     */
    void retire(SamplePtr oldSample) noexcept {
        if (!oldSample) {
            return;
        }
        if (!lockFreeRetireQueue_.tryPushMove(std::move(oldSample))) {
            std::lock_guard<std::mutex> lock(poolMutex_);
            retiredPool_.push_back(std::move(oldSample));
        }
    }

    /**
     * @brief Invoked periodically (e.g., 10 Hz timer on the Message Thread).
     *        Deallocates any retired SpectralSampleData whose use_count() == 1
     *        (i.e. no longer referenced by the Audio Thread or any voice).
     * @return Number of SpectralSampleData instances freed in this pass.
     */
    std::size_t collectGarbage() {
        std::lock_guard<std::mutex> lock(poolMutex_);

        // Drain lock-free retirement queue into the managed pool
        SamplePtr incoming;
        while (lockFreeRetireQueue_.tryPop(incoming)) {
            if (incoming) {
                retiredPool_.push_back(std::move(incoming));
            }
        }

        std::size_t freedCount = 0;
        auto it = retiredPool_.begin();
        while (it != retiredPool_.end()) {
            if (!(*it) || it->use_count() <= 1) {
                it = retiredPool_.erase(it); // ~SpectralSampleData() runs safely on Message Thread
                ++freedCount;
            } else {
                ++it;
            }
        }
        return freedCount;
    }

    /**
     * @brief Returns the number of samples currently awaiting deferred reclamation.
     */
    [[nodiscard]] std::size_t pendingCount() const {
        std::lock_guard<std::mutex> lock(poolMutex_);
        return retiredPool_.size() + lockFreeRetireQueue_.sizeApprox();
    }

private:
    LockFreeSpscRingBuffer<SamplePtr, 64> lockFreeRetireQueue_;
    mutable std::mutex                    poolMutex_;
    std::vector<SamplePtr>                retiredPool_;
};

/**
 * @brief Lock-Free RCU Sample Slot for a single sample source S_i (i in 0..7).
 *
 * Uses a dual-slot RCU generation table + atomic active pointer so that:
 *   1. The Audio Thread acquires a pinned reference in wait-free O(1) time
 *      (zero mutexes, zero heap allocations, zero risk of calling free()).
 *   2. The Background Worker / Message Thread publishes newly analyzed samples
 *      and automatically hands the previous sample to DeferredSampleGarbageCollector.
 */
class alignas(kCacheLineBytes) SpectralSampleSlot {
public:
    using SamplePtr = std::shared_ptr<const SpectralSampleData>;

    /**
     * @brief RAII guard held by the Audio Thread during processBlock().
     *        Pins the active epoch so the pointer remains valid for the entire block.
     */
    class ScopedReadLease {
    public:
        ScopedReadLease() noexcept = default;

        explicit ScopedReadLease(const SpectralSampleSlot& slot) noexcept
            : slot_(&slot) {
            slot_->activeReaders_.fetch_add(1, std::memory_order_acquire);
            ptr_ = slot_->rawActivePtr_.load(std::memory_order_acquire);
        }

        ~ScopedReadLease() noexcept {
            if (slot_ != nullptr) {
                slot_->activeReaders_.fetch_sub(1, std::memory_order_release);
            }
        }

        ScopedReadLease(const ScopedReadLease&)            = delete;
        ScopedReadLease& operator=(const ScopedReadLease&) = delete;

        ScopedReadLease(ScopedReadLease&& other) noexcept
            : slot_(std::exchange(other.slot_, nullptr)),
              ptr_(std::exchange(other.ptr_, nullptr)) {}

        ScopedReadLease& operator=(ScopedReadLease&& other) noexcept {
            if (this != &other) {
                if (slot_ != nullptr) {
                    slot_->activeReaders_.fetch_sub(1, std::memory_order_release);
                }
                slot_ = std::exchange(other.slot_, nullptr);
                ptr_  = std::exchange(other.ptr_, nullptr);
            }
            return *this;
        }

        [[nodiscard]] const SpectralSampleData* get() const noexcept { return ptr_; }
        [[nodiscard]] const SpectralSampleData& operator*() const noexcept { return *ptr_; }
        [[nodiscard]] const SpectralSampleData* operator->() const noexcept { return ptr_; }
        [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }

    private:
        const SpectralSampleSlot* slot_{nullptr};
        const SpectralSampleData* ptr_{nullptr};
    };

    SpectralSampleSlot() = default;
    SpectralSampleSlot(const SpectralSampleSlot&)            = delete;
    SpectralSampleSlot& operator=(const SpectralSampleSlot&) = delete;

    /**
     * @brief Wait-free acquisition of the sample on the Real-Time Audio Thread.
     */
    [[nodiscard]] ScopedReadLease acquireReadLease() const noexcept {
        return ScopedReadLease(*this);
    }

    /**
     * @brief Acquires a shared_ptr snapshot for non-real-time threads (UI / Rule Preview)
     *        or when shared ownership is desired while GC keeps use_count() >= 2.
     */
    [[nodiscard]] SamplePtr acquireSharedSnapshot() const noexcept {
        std::lock_guard<std::mutex> lock(writerMutex_);
        return currentOwner_;
    }

    /**
     * @brief Publishes a newly analyzed SpectralSampleData from the Offline Worker / Message Thread.
     *        The displaced sample is transferred to the DeferredSampleGarbageCollector so it is
     *        never freed while the Audio Thread may still be reading it.
     */
    void publishSample(SamplePtr newSample, DeferredSampleGarbageCollector& gc) noexcept {
        SamplePtr oldSample;
        {
            std::lock_guard<std::mutex> lock(writerMutex_);
            // Always keep a backup reference in the GC BEFORE swapping so use_count() >= 2
            oldSample     = std::move(currentOwner_);
            currentOwner_ = std::move(newSample);
            rawActivePtr_.store(currentOwner_.get(), std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_acq_rel);
        }

        if (oldSample) {
            gc.retire(std::move(oldSample));
        }
    }

    /**
     * @brief Clears the slot and retires its sample to the DeferredSampleGarbageCollector.
     */
    void clearSlot(DeferredSampleGarbageCollector& gc) noexcept {
        publishSample(nullptr, gc);
    }

    [[nodiscard]] bool isLoaded() const noexcept {
        return rawActivePtr_.load(std::memory_order_acquire) != nullptr;
    }

    [[nodiscard]] std::uint64_t generation() const noexcept {
        return generation_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::uint32_t activeReaderCount() const noexcept {
        return activeReaders_.load(std::memory_order_acquire);
    }

private:
    alignas(kCacheLineBytes) std::atomic<const SpectralSampleData*> rawActivePtr_{nullptr};
    alignas(kCacheLineBytes) mutable std::atomic<std::uint32_t>     activeReaders_{0};
    alignas(kCacheLineBytes) std::atomic<std::uint64_t>             generation_{0};

    mutable std::mutex writerMutex_;
    SamplePtr          currentOwner_;
};

} // namespace spectile::core

namespace spectile::dsp {
using spectile::core::DeferredSampleGarbageCollector;
using spectile::core::LockFreeSpscRingBuffer;
using spectile::core::LockFreeTripleBuffer;
using spectile::core::SpectralSampleSlot;
} // namespace spectile::dsp
