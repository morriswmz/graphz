#include <cstddef>
#include <stdexcept>

namespace graphz
{
    /**
     * Provides read/write access to a chunk of consecutive memory (usually
     * part of an array). This small wrapper class exists to avoid using too
     * many raw pointers. This class is NOT responsible for the underlying
     * memory management.
     */
    template <typename T>
    class ArraySlice
    {
    public:
        using size_type = std::size_t;
        using value_type = T;

        ArraySlice(): ptr_(nullptr), size_(0) {}
        ArraySlice(T *ptr, const size_type size): ptr_(ptr), size_(size) {}

        const size_type size() const { return size_; }

        value_type& operator[](size_type i) { return ptr_[i]; }
        const value_type& operator[](size_type i) const { return ptr_[i]; }

        ArraySlice slice(const size_type start, const size_type len)
        {
            EnsureSliceRange(start, len);
            return ArraySlice(ptr_ + start, len);
        }

        const ArraySlice Slice(const size_type start, const size_type len) const
        {
            EnsureSliceRange(start, len);
            return ArraySlice(ptr_ + start, len);
        }

    private:
        void EnsureSliceRange(const size_type start, const size_type len) const
        {
            if (size_ - start < len)
            {
                throw std::out_of_range("Sub slice out of range.");
            }
        }

        value_type *ptr_;
        size_type size_;
    };
}
