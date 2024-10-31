# 字符串

## KMP 

	 模式匹配  
	 给定一个长度为n得字符串S,查找长度为m的字符串T  
	 母串:S,模式串:T  
	 朴素匹配算法:暴力法,从左往右试  
	 S: a, b, c, x, y, z, 1, 2, 3  
	 T: 1, 2, 3

	 KMP算法:  
	 先前的做法:如果不匹配,i回到原来的起点+1的位置,j回到0  
	 但是: 失配点之前各个字母不同,i不需要动,j回到0即可  
	 
	 如果有部分字母相同?        i不同,j如何变?  
	 
	 只需求出T数组每一个前缀的最长公共前后缀的长度(NEXT数组)  
	 Next[i]存储T[0] - T[1]的最长公共前后缀(不含本身)  
	 已知Next[1] - Next[i],如何求Next[i+1]
	 
	 只要不匹配,j=Next[i]

### 斤斤计较的小Z
http://www.lanqiao.cn/problems/2047/learning/

```python
# 母串S,模式串T
Next = [0] * 1000010
# 1.求Next数组
def get_next(T):
    # 求Next[i]
    for i in range(1, len(T)):
        j = Next[i]
        # 不断地往前找到能够匹配的j
        while j > 0 and T[i] != T[j]:
            j = Next[j]
        if T[i] == T[j]:
            Next[i + 1] = j + 1
        else:
            Next[i + 1] = 0

# 返回字符串s中t出现的次数
def KMP(s, t):
    get_next(t)
    ans = 0
    j = 0
    for i in range(len(s)):
        # 每次维护s[i]和t[j]
        while j > 0 and s[i] != t[j]:
            j = Next[j]
        # 判断是否匹配上
        if s[i] == t[j]:
            j += 1
        # 判断是否匹配完成
        if j == len(t):
            ans += 1
            j = Next[j] # 二次匹配
    return ans

t = input()
s = input()
print(KMP(s, t))
```

## 字符串哈希

	 字符串哈希  
	 把字符串当作一个数字,降低字符串匹配的复杂度  
	 数字相同就表示字符串相同  
	   进制转换+求余即可  
	 进制选择: 根据可用的字母数量选择  
	 余数选择: 一般选择大质数  
	 
	 只要哈希结果相同视为字符串相同  
	 如何利用字符串哈希完成字符串匹配  
	 给定一个长度为n的字符串S,查找长度为m的字符串T  
	 对于字符串T而言,可以映射为一个整数numT  
	 对于字符串S,我们采用前缀哈希快速求出每个长度为m的子串的哈希值  
 
	 图:
	   
	 要求'bcd': Hash[3] - hash[0]*26*26*26  
	 要求'cd': Hash[3] - hash[1]*26*26

![](https://img-blog.csdnimg.cn/direct/ec47c3195ede44639960d27900530d95.png)

```python
t = input()
s = input()
m, n = len(t), len(s)
# 把字符串S的哈希前缀求出来
B = 26
mod = 1000000007
# 哈希前缀
hash = [0] * (n + 1)
for i in range(1, n + 1):
    hash[i] = hash[i - 1] * B + ord(s[i - 1]) - ord('A')
    hash[i] %= mod
# 匹配所有区间
# [0, m-1],[1, m+1],[2, m+2]
# 每次匹配s[l...l+m-1]的哈希值是否等于T的哈希值
numT = 0
for c in t:
    numT = numT * B + ord(c) - ord('A')
    numT %= mod

p = (B ** m) % mod
ans = 0
# 枚举s的所有区间,判断是否等于numT
for l in range(1, n+1):
    r = l + m - 1
    if r > n:
        break
    # 求s[l...r]的哈希值 = hash[r] - hash[l - 1] *(B ** m)
    if (hash[r] - hash[l - 1] * p % mod + mod) % mod == numT:
        ans += 1

print(ans)

```

## Manacher算法

	 最长回文子串问题  
	 给定一个长度为n的字符串s,求最长回文子串  
	   回文串: 从左往右、从右往左均相同  
	   子串: 连续的一部分  
	 暴力做法: 枚举所有子串,逐个判断O(n³)  
	 枚举中心点,向两端扩展 O(n²)# 二分+哈希: 枚举中心点,向两端扩展,扩展时二分区间长度,利用哈希来判断是否相等 O(nlog(n))# Manacher算法: O(n)  
  
	 d[i]表示以i为中心可扩展的长度  
	 偶数长度回文串没有中心如何处理:  
	   插入符号,将其转换成奇数长度:字符之间以#分割  
	   abaaba => #a#b#a#a#b#a#  
	 维护一个最右回文子串[left, center, right]  
  
	 核心思想:  
	 更新dp时,同时更新center和right  
	 最终答案恰好为dp数组的最大值  
	 dp记录的是可扩展的最大值:  
	   对于字母位置而言,最终长度为dp * 2 + 1, 但是有dp + 1 个#号(#a#b#a#)  
	   对于#而言,最终为dp * 2 + 1个,也有dp + 1 个#号(#a#b#b#a#)

![](https://img-blog.csdnimg.cn/direct/3343a336e42d4d24be29b276a15b01f7.png)

```python
def expend(s, l, r):
    # 保证s[l] == s[r]即可
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1
        r += 1
    # 最终返回
    return (r - l - 2) // 2


def longestpalindrome(s):
    s = '#' + '#'.join(list(s)) + '#'
    # 构建dp数组
    dp = [0] * len(s)
    # 维护最右端的回文子串
    center, right = 0, 0

    #求dp[i]
    for i in range(1, len(s)):
        if i > right:
            dp[i] = expend(s, i, i)
        else:
            # 对称点
            i_sym = 2 * center - i
            # 当前能利用的最长回文区间
            min_len = min(dp[i_sym], right - i)
            dp[i] = expend(s, i - min_len, i + min_len)

        if i + dp[i] > right:
            center = i
            right = i + dp[i]

    print(s)
    print(dp)
    print(max(dp))
#输入字符串
s = input()
longestpalindrome(s)
```

## 01-Tire 字典树
