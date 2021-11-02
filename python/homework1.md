## Python编程基础

### 1. 输入输出学习

输入两个整数，并计算二者的加减乘除的结果


```python
a = input('请输入第一个整数: ')
b = input('请输入第二个整数: ')

# 分别把 a、b 转换成整数
def fun2Int(a):
    try:
        a = int(a)
        return a
    except Exception as e:
        # print(e)
        print(f"输入{a}错误!")
        flag = True
        while flag:
            a = input('请输整数: ')
            a = fun2Int(a)
            if type(a) == int:
                flag = False
            print("")
            return a

# print(fun2Int(a))
# print(fun2Int(b))
a = fun2Int(a)
b = fun2Int(b)
print(f"输入a={a},输入b={b}")
# 计算 a、b 的和、差、积、商，赋值给变量c
def dev(a ,b):
    try:
        assert b!=0
        c = a/b
        return c
    except AssertionError as e:
        print("除数b=0,请检查！",end='')    
        return None

# 打印c
print(f"c=a+b={a+b} ")
print(f"c=a-b={a-b} ")
print(f"c=a*b={a*b} ")
print(f"c=a/b={dev(a,b)} ")
```

    请输入第一个整数: 12
    请输入第二个整数: 5
    输入a=12,输入b=5
    c=a+b=17 
    c=a-b=7 
    c=a*b=60 
    c=a/b=2.4 


### 2. 条件判断

输入两个整数，如果两个整数之和小于100，则输出 '小于100'，否则输出 '不小于100'


```python
a = input('请输入第一个整数: ')
b = input('请输入第二个整数: ')

# 分别把 a、b 转换成整数
a = fun2Int(a)
b = fun2Int(b)

# 计算 a、b 的和，赋值给变量c
c =a + b
print(f"输入a={a},输入b={b}")
# 判断c是否小于100，按要求输出
if c<100:
    print("小于100")
else:
    print("不小于100")
```

    请输入第一个整数: 56
    请输入第二个整数: 99
    输入a=56,输入b=99
    不小于100


### 3. 列表学习（数组）

1.创建一个含有元素1，2，4，8，16，32的列表

尽可能的**写多种**的方法实现（两种以上该问满分）

2.分别完成以下操作（在原列表基础上）

·  输出第2个元素   
·  删除第2个元素   
·  更改第2个元素为0   


```python
# 列表统一命名为L
# 方法一：
L = [2 ** i for i in range(6)]
print(L)
# 方法二：
L = [1 if i ==0 else 2 << i-1 for i in range(6)  ]
print(L)

print(L[1])
L.remove(2)
print(L)
L.insert(1,0)
print(L)

```

    [1, 2, 4, 8, 16, 32]
    [1, 2, 4, 8, 16, 32]
    2
    [1, 4, 8, 16, 32]
    [1, 0, 4, 8, 16, 32]


### 4. 斐波那契数列

**资源限制**

时间限制：1.0s 内存限制：256.0MB

想要拿满分的话，资源限制一定要特别注意！

**问题描述**

Fibonacci数列的递推公式为：$F_n$ = $F_(n−1)$ + $F_(n−2)$ 其中$F_1$=$F_{2}$ = 1 

当n比较大时，$F_n$也非常大，现在我们想知道，$F_n$除以10007的余数是多少。

**输入格式**

输入包含一个整数n。

**输出格式**

输出一行，包含一个整数，表示Fn除以10007的余数。


```python
n = int(input())
import time
start = time.time()
#递归时间不满足
# def feb(n):
#     if n ==0:
#         return 0
#     if n <= 2:
#         return 1
#     else:
#         return feb(n-1)+feb(n-2)
def feb(n):
    if n == 0:
        return 0
    if n > 0 and n <= 2:
        return 1
    m1 = 1
    m2 = 1
    res = 0
    i = 1
    while i <n:
        res = m1 + m2
        m1 =m2
        m2 =res
        i = i+1
    return res

result = feb(n)%10007
end = time.time()
print(end-start)
print(result)
```

    1000
    0.0008945465087890625
    1115

