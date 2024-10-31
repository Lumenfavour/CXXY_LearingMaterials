# 一、DFS基础

	搜索算法: 穷举问题解空间部分/所有情况,从而求出问题的解  
	深度优先搜索:  
		  1.本质上是暴力枚举  
		  2.深度优先:尽可能一条路走到底，走不了再回退  
  
	给定一个数字x,将其拆分成3个正整数,后一个要求大于等于前一个,给出方案.  
		  1.最简单的思想:三重循环暴力求解  
		  2.拆分成4个正整数？  
		  3.拆分成n个正整数？  
		  4.就需要实现n重循环  
		  5.n重循环=特定的树状结构=DFS搜索  

	1.DFS和n重循环  
	2.给定一个数字x=6,将其拆分成3个正整数,后一个要求大于等于前一个,给出方案.  
	3.n重循环: n层的树  
	4.DFS:从上往下找一条合法的路径(路径值不递减、长度为n,和为x)


```python模板
def dfs(depth, last_value):  
    # depth: 表示当前处于第depth层  
    # 递归出口  
    if depth == n:        
	    for i in range(1, n):            
		    if path[i] >= path[i - 1]:                
		    continue            
	    else:               
		     return        
     if sum(path) != x:            
	     return        
     # 答案  
    print(path)        
    return  
    # 对于每一层,枚举当前拆出的数字  
	for i in range(last_value, x + 1):      
		path[depth] = i        
		dfs(depth + 1, i)  
		
x = int(input())  
n = int(input())  
# path[i]表示第i个数字  
path = [0] * n  
dfs(0, 1)
```

### 分糖果  
http://www.lanqiao.cn/problems/4124/learning/  
  
>七重循环，每重循环1个小朋友  
   每个小朋友枚举所有的糖果情况

#### 法一
```python
# ans表示方案数
ans = 0
def dfs(depth, n, m):
    # depth:第几个小朋友
    # n: 第一种糖果剩余量
    # m: 第二种糖果剩余量

    # 当分完所有小朋友后保证手上没有糖果
    if depth == 7:
        if n == 0 and m == 0:
            global ans
            ans += 1
        return

    # 枚举当前小朋友的糖果可能性
    # 枚举第一种糖果
    for i in range(0,6):
        # 枚举第二种糖果
        for j in range(0, 6):
            # 第depth个小朋友有i个第一种，j个第二种
            if 2 <= i + j <= 5 and i <= n and j <= m:
                dfs(depth + 1, n - i, m - j)

dfs(0, 9, 16)
print(ans)
```
#### 法二
```python
# ans表示方案数
ans = 0
path = [[0,0] for i in range(7)]

def dfs(depth):
    # depth:第几个小朋友

    # 当分完所有小朋友后保证手上没有糖果
    if depth == 7:
        sum1, sum2 = 0, 9
        for i in range(7):
            sum1 += path[i][0]
            sum2 += path[i][1]
        if sum1 == 9 and sum2 == 16:
            global ans
            ans += 1
        return

    for i in range(0, 6):
        for j in range(0, 6):
            if 2 <= i + j <= 5:
                path[depth][0] = i
                path[depth][1] = j
                dfs(depth + 1)
dfs(0, 9, 16)  
print(ans)
```
##  买瓜
http://www.lanqiao.cn/problems/3505/learning/

	 N重循环,每重循环三种情况:  
		 1.买一个  
		 2.买一半  
		 3.不买

```python
def dfs(depth,weight,cnt):
    # depth:第depth个瓜
    # weight: 表示买到的瓜的重量
    # cnt: 表示当前劈的次数

    # 剪枝
    if weight > m:
        return
    if weight == m:
        global ans
        ans = min(ans, cnt)
    if depth == n:
        return

    if depth == n:
        return
    # 枚举当前瓜的三种情况
    # 不买
    dfs(depth + 1, weight + 0, cnt)
    # 买
    dfs(depth + 1, weight + A[depth], cnt)
    # 买一半
    dfs(depth + 1, weight + A[depth]//2, cnt + 1)


n ,m = map(int, input().split())
m *= 2
A = list(map(int, input().split()))
A = [x * 2 for x in A]
ans = n+1
dfs(0, 0, 0)
if ans == n + 1:
    ans = -1
print(ans)
# 只能过65%的例子 需要更多的剪枝优化
```


# 二、DFS回溯
	1.回溯: 就是DFS的一种，在搜索尝试过程中寻找问题的解,当发现已不满足求解条件时,就"回溯"返回,尝试别的路径  
	2.回溯更强调: 此路不通,另寻他路,走过的路需要打标记  
	3.回溯法一般在DFS的基础上加上一些剪枝策略  
  
	回溯模板——求排列  
		1.排列要求数字不重复——每次选择的数字需要打标记——vis数组  
		2.要输出当前排列——记录路径——path数组  
		3.回溯: 先打标记、记录路径、然后下一层、回到上一层、清除标记

##  排列树
```python
def dfs(depth):
    # depth:第depth个数字
    if depth == n:
        print(path)
        return True

    # 第depth个数字可以从1-n进行选择
    for i in range(1, n + 1):
        # 选择的数字必须未标记
        if vis[i]:
            continue
        vis[i] = True
        path.append(i)
        dfs(depth + 1)

        vis[i] = False
        path.pop(-1)

n = int(input())
vis = [False] * (n + 1)
path = [ ]
dfs(0)
```

## 求子集
```python
n = int(input())
a = list(map(int, input().split()))

path = [ ]
def dfs(depth):
    if depth == n:
        print(path)
        return
    # 选
    path.append(a[depth])
    dfs(depth + 1)
    path.pop(-1)

    # 不选
    dfs(depth + 1)

dfs(0)
```

### N皇后
http://www.lanqiao.cn/problems/1508/learning/

	 dfs枚举每一行放置的列  
	 标记:  
		1.每列只能放一个  
		2.主对角线: x + y  
		3.副对角线: x - y + n

```python
def dfs(x):
    # 第x层的皇后
    if x == n + 1:
        global ans
        ans += 1
        return



    # 第x层的皇后枚举每一列
    for y in range(1, n + 1):
        # 当前的坐标为(x,y),要求当前列,当前主对角线、副对角线不能被攻击到
        # if vis1[y] is False and vis2[x + y] is False and vis3[x - y + n] if False:
        if vis1[y] or vis2[x + y] or vis3[x - y + n] :
            continue
        # 此时(x,y)是一个合法点
        # 打标记、进入下一层搜索、回到该层、
        vis1[y] = vis2[x + y] = vis3[x - y +n] = True
        dfs(x + 1)
        vis1[y] = vis2[x +y] = vis3[x - y +n] = False


n = int(input())
vis1 = [False] * (n+1)
vis2 = [False] * (2 * n + 1)
vis3 = [False] * (2 * n + 1)

ans = 0
dfs(1)
print(ans)
```

### 小朋友崇拜圈
http://www.lanqiao.cn/problems/182/learning/

```python
# 当前位于点x,步长为length
import sys
# 扩栈:递归层数过大,需要设置最大栈空间
sys.setrecursionlimit(10**6)
def dfs(x, length):
    # 记录走到x的步长为length
    vis[x] = length

    # 接下来要走下一个点
    # 判断下一个点是否走过
    if vis[a[x]] != 0:
        global ans
        ans = max(ans, length -vis[a[x]] + 1)
    else:
        dfs(a[x], length + 1)

n = int(input())
a = [0] + list(map(int,input().split()))
# 标记数组: vis[x] 表示点x的步数
vis = [0] * (n + 1)
ans = 0

for i in range(1, n+1):
    # 对于每个单独的连通块,都要做一遍dfs
    if vis[i] == 0:
        dfs(i, 1)
```


### 全球变暖
http://www.lanqiao.cn/problems/178/learning/

	 从左到右,从上到下,对于每一个未标记的陆地,做一遍DFS  
	 DFS目的:扩展能够到达所有点,并打上标记  
	 可以统计出有多少个连通块

```python
  
"""  
.......  
.##....  
.##....  
....##.  
..####.  
...###.  
.......  
"""  
  
import sys  
# 扩栈:递归层数过大,需要设置最大栈空间  
sys.setrecursionlimit(10**6)  
def dfs(x, y):  
    # 当前处于(x, y)  
    vis[x][y] = 1  
  
    #判断当前点是否为高地  
    if Map[x][y + 1] == '#' and Map[x + 1][y] == '#' and Map[x-1][y] == '#' and Map[x][y -1] == '#':  
        global flag  
        flag = True  
  
    # 往四周扩展,把所有相邻的点打上标记  
    for (delta_x,delta_y) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:  
        xx, yy = x + delta_x, y + delta_y  
        # 判断: 新坐标能走吗？  
        if Map[xx][yy] == '#' and vis[xx][yy] == 0:  
            dfs(xx, yy)  
  
N = int(input())  
Map = []  
vis = []  
for i in range(N):  
    Map.append(list(input()))  
    vis.append([0] * N)  
  
# ans 表示淹没岛屿的数量  
ans = 0  
  
for i in range(N):  
    for j in range(N):  
        if Map[i][j] == '#' and vis[i][j] == 0:  
            # flag表示当前岛屿是否存在  
            flag = 0  
            dfs(i, j)  
            if flag == False:  
                ans += 1  
  
print(ans)
```

# 三、DFS剪枝

在搜索过程中，如果需要完全遍历所有情况可能需要很多时间  
在搜索到某种状态时，根据当前状态判断出后续无解,则该状态无需继续深入搜索  
例如:给定N个正整数,求出有多少个子集之和小于等于K  
	  在搜索过程中当前选择的数字和已经超过K则不需要继续搜索  
  
>可行性剪枝:  当前状态和题意不符,并且往后的所有情况和题意都不符,那么就可以进行剪枝  
>最优性剪枝:  在搜索过程中,当前的状态已经不如已经找到了最优解,也可以剪枝，不需要继续搜索  
### 数字王国之军训排队  
http://www.lanqiao.cn/problems/2942/learning/  

	DFS搜索,枚举每个学生分到每个组内  
	可行性剪枝: 要满足题目条件  
	最优性剪枝: 判断当前状态是否比ans更劣

```python
def check(x, group):
    # 判断学生x能否加入group中
    for y in group:
        if x % y == 0 or y % x == 0:
            return False
    return True


def dfs_pruning(depth):
    # 最优性剪枝
    global ans
    # 如果当前分组状态已经比ans大,则该分组策略肯定不行
    if len(Groups) > ans:
        return

    # depth: 当前是第几个学生
    # 递归出口
    if depth == n:
        global ans
        ans = min(ans, len(Groups))
        return

    # 对于每个学生，枚举该学生放在哪一组
    # 遍历每一组
    for every_group in Groups:
        # 当前第depth个学生呢个否加入当前组every_group
        if check(a[depth], every_group):
            every_group.append(a[depth])
            dfs_pruning(depth + 1)
            every_group.pop()
    # 对于每个学生,也可以单独作为一组
    Groups.append([a[depth]])
    dfs_pruning(depth + 1)
    Groups.pop()


n = int(input())
a = list(map(int, input().split()))
# Group表示分组情况，每个元素表示一个组内的情况
Groups = [ ]
ans = n
dfs_pruning(0)
print(ans)
```

### 特殊的多边形
http://www.lanqiao.cn/problems/3075/learning/

	先考虑简单版:乘积为v有多少种n边形  
	DFS处理出所有乘积对应的所有可能  
		维护一个递增的边长序列(唯一性)  
		枚举第i边的长度,最小最大范围(剪枝)  
		最终check是否满足N边形  
			最小的任意N-1条边之和大于第N边  
		预处理+前缀和O(1)查询答案  
  
 >1.  利用DFS求所有的N边形,边长乘积不超过10000  
 >2.  N边形: 最小的N-1条边之和 > 第N边  
 >3.  <=> N边之和 > 2 * 第N边    sum > 2 * path[-1].

```python
def dfs_pruning(depth, last_val, tot, mul):
    """
    :param depth:       第几条边
    :param last_val:    上一条边的边长
    :param tot:         当前所有边长之和
    :param mul:         当前所有边长之积
    :return:
    """
    if depth == n:
        #N边形的条件

        if tot > 2 * path[-1]:
            # 此时是一个合法的N边形
            ans[mul] += 1
        return

    # 枚举第depth条边的边长
    for i in range(last_val + 1, 100000):
        # 最优性剪枝: 要保证乘积不超过100000
        # 先前选择了depth个数字,乘积为mul
        # 后续还有n-depth个数字,每个数字都要 > i
        if mul * (i **(n - depth)) <= 100000:

            path.append(i)
            dfs_pruning(depth + 1, i, tot + i, mul * i)
            path.pop()
        else:
            break

# ans[i]表示价值为i的N边形数目
ans = [0] * 1000001

t, n = map(int, input().split())
path = []
dfs_pruning(0, 0, 0, 1)
# 每次询问一个区间l, r输出有多少个N边形的价值在[l,r]中
# 等价于ans[l]+...+ans[r], 所以需要对ans求前缀和
for i in range(100001):
    ans[i] += ans[i - 1]
for _ in range(t):
    l, r = map(int, input().split())
```

# 四、记忆化搜索
记忆化: 通过记录已经遍历过的状态的信息,从而避免对同一状态重复遍历的搜索实现方式  
记忆化=dfs+额外字典 

>1.如果先前已经搜索过:直接查字典,返回字典中结果  
   2.如果先前没有搜索过:继续搜索,最终将该状态结果记录到字典中  
  
	  1.斐波那契数列: 设F[0]=1, F[1]=1, F[n]=F[n-1]+F[n-2].  
	  2.求F[n] 结果对 le9+7 取模  
	  3.每次搜索时将当前状态答案记录到字典中  
	  4.后续搜索直接返回结果  
	  5.直接递归求解: 存在大量重复计算项

```python
import sys
sys.setrecursionlimit(100000)
dic = {0:1, 1:1}
def f(n):
    if n in dic.keys():
        return dic[n]
    dic[n] = (f(n-1) + f(n-2)) % 100000007
    return dic[n]

n = int(input())
print(f(n))
```

```python
from functools import lru_cache
@lru_cache(maxsize=None)
def f(x):
    if x == 0 or x == 1:
        return 1
    return f(x-1) + f(x-2)

n = int(input())
print(f(n))
```

### 混沌之地
http://www.lanqiao.cn/problems/3820/learning/

	 1.走到(x,y),z 表示是否使用喷气背包  
	 2.当x,y,z固定时,具有唯一解,因此可以使用记忆化搜索  
	 3.时间复杂度<x,y,z> 1000*1000*2

```python
from functools import lru_cache
@lru_cache(maxsize=None)
def dfs(x,y,z):
    """
    坐标为(x,y),z表示是否使用过背包
    :param x:
    :param y:
    :param z:
    :return: 返回True表示能够逃离，返回False表示不能逃离
    """
    if x == C and y == D:
        return True

    # 如果没走到终点，那就四个方向判断
    for delta_x, delta_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        xx, yy = x + delta_x, y + delta_y
        # 新坐标: 不能越界
        if xx < 0 or xx >= n or yy < 0 or yy >= m:
            continue
        if Map[xx][yy] < Map[x][y]:
            if dfs(xx, yy, z):
                return True

        # 在(x,y)处使用喷气背包
        elif z == False and Map[xx][yy] < Map[x][y] + k:
            if dfs(xx, yy, True):
                return True

    return False

n, m, k = map(int, input().split())
A, B, C, D = map(int, input().split())
A, B, C, D = A - 1, B - 1, C - 1, D - 1
Map = []

for i in range(n):
    Map.append(list(map(int, input().split())))
if dfs(A, B, False):
    print("Yes")
else:
    print("No")
```

### 地宫取宝
https://www.lanqiao.cn/problems/216/learning/

	1.从(x,y)出发,先前已经有宝物z件,已有的最大宝物价值为w的方案数记为dfs(x,y,z,w) 
	2.只要确定四元组,就确定当前的方案数  
	3.答案 = dfs(1,1,0,-1)# 最终点=dfs(n,m,z,?) or dfs(n,m,z-1,?)  
	4.(x,y)处可选,可不选,然后可以往右走或者往下走

```python
from functools import lru_cache
@lru_cache(maxsize=None)
def dfs(x, y, z, w):
    """

    :param x: 当前的横坐标
    :param y: 当前的纵坐标
    :param z: 先前拿的宝物数量
    :param w: 先前拿的宝物的最大价值
    :return:  当前状态出发的方案数
    """
    # 递归出口
    if x == n - 1 and y == m -1 :
        if z == k:
            return 1
        if z == k - 1 and w < Map[x][y]:
            return 1
        return 0

    # 方案数=右边的方案数+下边的方案数,两个方向
    ans = 0
    for delta_x, delta_y in [(1,0),(0,1)]:
        xx, yy = x + delta_x, y + delta_y
        if xx < n and yy < m:
            #当前不选择宝物,走到(xx,yy)
            ans ++ dfs(xx, yy, z, w)
            # 当前选择宝物,然后再走到(xx,yy)
            if w < Map[x][y]:
                ans += dfs(xx, yy, z + 1,Map[x][y])

            ans %= 1000000007
    return ans
n, m, k = map(int, input().split())
Map = []
for i in range(n):
    Map.append(list(map(int, input().split())))
print(dfs(0, 0, 0, -1))
```

