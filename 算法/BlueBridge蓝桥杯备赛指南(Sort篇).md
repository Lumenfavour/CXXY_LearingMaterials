# 一、基础排序算法
## 1.冒泡排序

	算法步骤  
		1.比较相邻元素，如果第一个大于第二个则交换  
		2.从左往右遍历一遍，重复第一步，可以保证最大的元素在最后面  
		3.重复上述操作，可以得到第二大、第三大、...  
  
	给定一个长度为n的列表，算法循环n-1次可以得到有序序列  
		第一次循环两两比较: <a[0],a[1]>,...,<a[n-4],a[n-3]>,<a[n-3],a[n-2]>,<a[n-2],a[n-1]>  
		第二次循环两两比较: <a[0],a[1]>,...,<a[n-4],a[n-3]>,<a[n-3],a[n-2]>  
		第三次循环两两比较: <a[0],a[1]>,...,<a[n-4],a[n-3]>  
		第i次循环两两比较: <a[0],a[1]>,...,<a[n-i-1],a[n-i]>  
		第n-1次循环两两比较: <a[0],a[1]>
	
>时间复杂度: O(n^2),空间复杂度O(1),稳定
```python
n = int(input())  
a = list(map(int, input().split()))  
# 循环n-1次，每次获得第i大  
for i in range(1,n):  
    # 每一次比较a[j] 和 a[j+1]    for j in range(0,n-i):  
        if a[j] > a[j+1]:  
            a[j],a[j+1] = a[j+1],a[j]  
  
print(' '.join(map(str, a)))
```
## 2.选择排序
	算法步骤  
		1.从左往右找到最小的元素，放在起始位置  
		2.重复上述步骤，依次找到第2小、第3小的元素...  
  
	给定一个长度为n的列表，算法循环n-1次可以得到有序序列  
		第0次循环从[0,n-1]中找最小元素a[x],与a[0]交换  
		第1次循环从[1,n-1]中找最小元素，与a[1]交换  
		第2次循环从[2,n-1]中找最小元素，与a[2]交换  
		第i次循环从[i,n-1]中找最小元素，与a[i]交换  
		第n-2次循环从[n-2,n-1]中找最小元素，与a[n-2]交换  
	
>时间复杂度: O(n^2),空间复杂度O(1),稳定
```python
n = int(input())  
a = list(map(int, input().split()))  
  
for i in range(n-1):  
    # 第i次从[i,n-1]找最小值  
    min_value = a[i]  
    min_idx = i  
    for j in range(i,n):  
        if a[j] < min_value:  
            min_value = a[j]  
            min_idx = j  
    # 将最小值和最前面的元素交换  
    a[i], a[min_idx] = a[min_idx],a[i]  
  
print(' '.join(map(str,a)))
```
## 3.插入排序

	算法步骤  
		1.第一个元素看作已排序，从左往右遍历每个元素:  
		2.在已排序元素中从后往前扫描: 如果当前元素大于新元素，则该元素移动到后一位  
		3.重复第二步直至找到小于等于新元素则停止  
  
	将上述步骤看做摸牌，每摸一张牌从后往前判断是否可以插入  
		对于第i张牌a[i],[0,i-1]中的牌都是已经排好顺序的  
		从后往前逐个判断a[j]是否大于a[i]  
		如果a[j]>a[i]: 则a[j]往后挪一个位置  
		如果a[j]<=a[i]: 此时在a[j+1]的位置放置a[i]  

>时间复杂度: O(n^2),空间复杂度O(1),不稳定

```python
n = int(input())  
a = list(map(int, input().split()))  
  
# 对于第i个数字，在区间[0,i-1]中从后往前找对应插入的位置  
for i in range(1, n):  
    value = a[i]  
    # 插入元素的下标  
    insert_idx = 0  
    for j in range(i-1, -1, -1):  
        if a[j] > value:  
            a[j+1] = a[j]  
        else:  
            insert_idx = j+1  
            break  
    # 插入第i个数字  
    a[insert_idx] = value  
  
print(' '.join(map(str, a)))
```
## 4.快速排序

	算法步骤  
		找一个基准值x  
		把列表分成3部分: 小于等于x的数字，x， 大于x的数字  
		左半部分和右半部分递归使用该策略  
  
	例: a = [3,5,8,1,2,9,4,7,6]  
	找到基准值3， [1.2], [3], [5,8,9,4,7,6]  
	左半部分 [1,2] 作为一个子问题求解  
	右半部分 [5,8,9,4,7,6] 作为一个子问题求解  
  
	例: a = [3,5,8,1,2,9,4,7,6] , left = 0, right = 8  
		设置基准值下标为left  
		存放小于等于基准值下标为 idx = left + 1  
		从left + 1 到 right 每个元素a[i] :  
			如果 a[i] <= 基准值: 则将 a[i], a[idx]互换, idx += 1  
		最终结果 [2,1], [3], [5,8,9,5,7,6]  
		左侧和右侧重复上述操作  
  
>时间复杂度: O(nlog(n)) ,空间复杂度O(nlog(n)), 不稳定

```python
def partition(arr, left, right):  
    """找一个基准值x，然后把数组分成三部分"""  
    # 基准值为a[left]  
    idx = left + 1  
    for i in range(left + 1, right + 1):  
        # 如果元素小于基准值，放到前面去  
        if arr[i] <= arr[left]:  
            arr[i], arr[idx] = arr[idx], arr[i]  
            idx += 1  
    # 把前半部分最后一个和基准值交换  
    arr[left], arr[idx-1] = arr[idx-1], arr[left]  
    return idx - 1  
  
# 对a[left,right]进行排序  
def quickSort(arr, left, right):  
    # print(arr, left, right)  
    if left < right:  
        mid = partition(arr, left, right)  
        # 此时a分成三部分: 【left,mid-1】 , 【mid】 , 【mid+1，right+1】  
        quickSort(arr, left, mid-1)  
        quickSort(arr, mid+1, right)  
  
n = int(input())  
a = list(map(int, input().split()))  
quickSort(a, 0, n-1)  
print(' '.join(map(str, a)))
```
## 5.归并排序

	算法步骤  
		1.先把数组分为两部分  
		2.每部分递归处理变成有序  
		3.将两个有序列表合并起来  
  
	首先考虑一个问题: 两个有序列表如何合并成一个列表  
	A=[1,3,5,6,7] 、 B=[2,3,4,9]  
	1.构建一个result = [ ]  
	2.当A非空且B非空  
		  比较A[0] 和 B[0]  
		  result添加较小的那个元素，并从原始数组弹出  
	3.如果A非空,把A添加到result末尾  
	4.如果B非空,把B添加到result末尾
	
```python
def Merge(A, B): 

    """  
    合并两个有序列表  
    :param A:    
    :param B:    
    :return:  
    """    
    
    result = []  
    while len(A) != 0 and len(B) != 0:  
        if A[0] <= B[0]:  
            result.append(A.pop(0))  
        else:  
            result.append(B.pop(0))  
  
    result.extend(A)  
    result.extend(B)  
    return result  
  
def MergeSort(A):  
    if len(A) <= 2:  
        return A  
    mid = len(A) // 2  
    # 分解成两部分，每部分递归处理  
    left = MergeSort(A[:mid])  
    right = MergeSort(A[mid:])  
    return Merge(left,right)  
n = int(input())  
a = list(map(int,input().split()))  
a = MergeSort(a)  
print(' '.join(map(str,MergeSort(a))))
```

## 6.桶排序

	利用函数映射关系，将输入数据分到有限的桶里，然后每个桶分别排序  
	  1.初始化K个桶  
	  2.遍历数据，将数据放入对应桶中  
	  3.每个桶单独排序  
	  4.各个桶的数据拼接起来  
  
	为了避免过多元素进入同一个桶  
		  1.优化映射方法，使得桶内元素均匀  
		  2.增加桶的数量，利用时间换空间  
		  
	数据如果服从均匀分布，则利用桶排序效率越高  
	桶内部排序算法可以采用其他简单的排序算法


```python
def Bucket_Sort(a, bucketcount):  
    """  
  
    :param a:    
    :param bucketcount:    
    :return:  
    """    
    minvalue, maxvalue = min(a), max(a)  
    # 桶大小: 每个桶的元素范围  
    bucketsize = (maxvalue - minvalue + 1) // bucketcount  
    res = [[] for _ in range(bucketcount + 1)]  
  
    #把所有元素放到对应桶中  
    for x in a:  
        # 元素x放在第几个桶  
        idx = (x - minvalue) // bucketsize  
        res[idx].append(x)  
  
    # 每个桶单独排序  
    ans = 0  
    for res_x in res:  
        res_x.sort()  
        ans += res_x  
    return ans  
  
n = int(input())  
a = list(map(int,input().split()))  
a = Bucket_Sort(a, min(10000, n))  
print(' '.join(map(str, a)))
```

