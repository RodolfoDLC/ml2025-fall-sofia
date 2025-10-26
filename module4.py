#!/usr/bin/env python3

"""
Simple program for module4:
- Read N (positive integer)
- Read N numbers (one by one)
- Read X and print the 1-based index of X among the numbers or -1 if not found
"""

def main():
    try:
        N = int(input().strip())
    except Exception:
        # invalid N
        return
    if N <= 0:
        return
    nums = []
    for i in range(N):
        try:
            v = int(input().strip())
        except Exception:
            return
        nums.append(v)
    try:
        X = int(input().strip())
    except Exception:
        return
    try:
        idx = nums.index(X) + 1
        print(idx)
    except ValueError:
        print(-1)

if __name__ == "__main__":
    main()
