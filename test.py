from typing import List
from collections import defaultdict
class Solution:
    def minMoves(self, nums: List[int], limit: int) -> int:
        n = len(nums)
        m = n // 2
        if n % 2 == 1: m + 1
        res = 0
        mp = defaultdict(int)
        for i in range(m):
            if nums[i] + nums[n - 1 - i] <= limit * 2: mp[nums[i] + nums[n - 1 - i]] += 1
            else: 
                if nums[i] >= limit and nums[n - 1 - i] >= limit: res += 1
        return m - max(mp.values()) + res
print(Solution().minMoves([1,2,2,1],2))