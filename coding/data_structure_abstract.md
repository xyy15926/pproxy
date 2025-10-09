---
title: Abstract Data Type
categories:
  - Algorithm
tags:
  - Algorithm
  - Data Structure
date: 2019-03-21 17:27:15
updated: 2025-09-25 11:05:27
toc: true
mathjax: true
comments: true
description: Abstract Data Type
---

##  概述

-   *Data Structure* 数据结构：数据（集）+ 关系
    -   存储结构：数据元素在计算机中的表示方式
        -   顺序存储（映像）：数据存储在连续的内存空间中
            -   初始化时即须确定容量，后续可能需要动态扩展
        -   链式（非顺序）存储（映像）：数据存储在不连续的内存空间中，通过指针相互连接
            -   此处指针指可表示连接关系的实体，可为指针、索引等
        -   顺序存储、链式存储可混用以满足需求，如：分块结构中块内顺序、块间链式
    -   逻辑结构：数据元素之间的逻辑关系
        -   逻辑关系的表示方式
            -   相对位置：仅适用于顺序存储结构
            -   指针（广义）：常用于链式存储结构，也可用于顺序存储结构中，如：数组中存储数组索引作为指针
        -   逻辑结构体现的逻辑关系才是核心，存储结构仅影响实现
            -   不同存储结构可具有相同的逻辑结构
            -   当然，不同的存储结构实现会影响存储效率、时间效率
        -   逻辑结构可分为 4 类
            -   集合：结构中数据元素只有同属一个集合的关系
            -   线性结构：结构中数据元素之间存在一对一关系
            -   树型结构：结构中数据元素存在一对多关系
            -   图状/网状结构：结构中数据元素存在多对多的关系

-   *Data Type* 数据类型：数据结构 + 操作
    -   逻辑结构决定、服务操作可行性，具体的存储结构（实现）决定操作效率
        -   即，逻辑结构本身对应基本的数据类型
        -   则，在基本数据类型上 **设定操作、修改存储结构** 即可得新数据类型
    -   根据数据类型值可分情况
        -   *Atomic Data Type*：原子类型，值不可分解（有计算机平台提供）
        -   *Fixed-Aggregate Data Type*：固定聚合类型，值由确定数目的成分按某种结构组成
        -   *Variable-Aggregate Data Type*：可变聚合类型，值的成分数目不确定
    -   *Abstract Data Type* 抽象数据类型：数据类型数学化、形式化抽象
        -   将数据类型行为和其实现相分离

-   思路、想法
    -   数据结构本身就可以认为是预处理，通过保存预处理信息保证效率

#   线性数据类型

-   线性结构：结构中数据元素之间存在一对一关系
    -   存在唯一一个被称为 “第一个” 的数据元素
    -   存在唯一一个被称为 “最后一个” 的数据元素
    -   除第一个外，每个数据元素均只有一个前驱
    -   除最后一个外，每个数据元素均只有一个后继

##  线性表

-   线性表：包含 $n$ 个数据元素的有限序列
    -   最简单的线性数据类型，未定义额外操作

### 顺序表

-   顺序表：顺序排列的数据域
    -   语言原生数据类型基本包含顺序表实现，提及场合不多
    -   操作、时间效率
        -   插入、删除：$O(n)$
        -   获取元素：$O(1)$

### 链表

```cpp
typedef struct LinkedListNode{
    int value;                                  // 数据域
    LinkedListNode * next;                      // 后继指针
} LinkedListNode;
```

![alg_linked_list_structure](imgs/alg_linked_list_structure.svg)

-   单向链表：数据域 + 指针域构成节点串联
    -   操作、时间效率
        -   插入、删除：$O(1)$
        -   获取元素：$O(n)$
    -   实现、应用技巧
        -   头节点、头指针：指向链表首个节点，不记录信息、方便调整链表

```cpp
typedef struct DoubleLinkedListNode{
    int value;                                  // 数据域
    DoubleLinkedListNode * next, * prev;        // 前驱、后继指针
} DoubleLinkedListNode;
```

![alg_double_linked_list_structure](imgs/alg_double_linked_list_structure.svg)

-   双向链表：数据域 + 前驱指针域 + 后继指针域构成节点串联
    -   双向异或链表：存储前驱、后继地址异或结果
        -   遍历时用前节点地址与之再次异或即得到后节点地址

-   其他一些链表数据结构
    -   循环链表：末尾节点指向首个节点成环
    -   双向循环链表

> - *OI Wiki* 链表：<https://oi-wiki.org/ds/linked-list/>

### 静态链表

```cpp
typedef struct StaticLinkedListNode{
    int value;
    int next;                           // 存储后继在数组索引
} StaticLinkedListNode, StaticLinkedList[MAXSIZE];
```

-   静态链表：数组描述链表 + 数组索引表示连接关系
    -   可在无指针语言中实现链表
    -   类似的，可用多个数组分别存储值、指针（索引）
        -   此实现甚至不要求语言支持自定义数据类型

```cpp
typedef int val[MAXSIZE];               // 多个数组分别存储值、指针
typedef int next[MAXSIZE];              // 语言支持随机访问序列即可
```

> - 索引作为指针表示连接关系同样可用于其他数据类型

### 技巧

-   双指针
    -   迭代操作注意事项
        -   保证退出循环
        -   所有值均被检验
        -   数据处理、移动指针的顺序
    -   相向双指针：两指针均向中间靠拢不断缩小搜索空间
        -   明确切片范围：是否包括端点，并尽量保持不变
        -   据此明确搜索空间范围，则搜索空间为空时结束循环
    -   滑动窗口：两指针同时向一侧移动，检验窗口内切片
        -   分类
            -   固定间隔双指针
            -   不同条件移动双指针
        -   示例
    -   快慢指针：两指针同时迭代、但运动速度不同
        -   示例
            -   判断单链表是否成环

-   端点利用
    -   末端点为空闲位置：存储有效值
        -   队列、栈插入元素：末端点处为空闲位置，可直接使用
        -   数组迭代比较：末端点处存储有效值，先比较再更新指针
    -   末端点为非空闲位置
        -   队列、栈：可以直接使用其末尾元素作为上次添加的元素，简化代码
    -   头指针/节点：添加链表头，保证特殊情况链表正常工作
        -   示例
            -   删除只包含一个元素的链表

-   两端限位器：在数据端点添加**标记**数据，便于
    -   指示“指针”已经到达数据端点
    -   简化特殊情况代码
        -   将循环中的判断条件合并
        -   端点标记数据符合一般情况，无需特殊处理
    -   示例
        -   数组末尾添加查找键（查询问题）：在顺序查找中可以将判断数组越界、查找键比较合并

##  *Stack*

![alg_stack_operation](imgs/alg_stack_operation.svg)

-   *Stack* 栈、*LIFO* 表：限定在栈顶（表尾）进行插入、删除的受限线性表
    -   栈按照 *LIFO* 先进后出原则运转，被限制在仅能在栈顶插入、删除数据
        -   入栈：栈顶插入数据
        -   出栈：栈顶删除数据
    -   实现递归嵌套的必须数据结构

-   单调栈：满足单调性的栈结构
    -   为保证栈单调性
        -   出栈：无区别
        -   入栈：弹出所有在插入新元素后破坏栈单调性元素后，在栈顶插入数据
    -   单调栈核心，即 **维护以当前元素为端点的最长有序序列**

> - *OI Wiki* 栈：<https://oi-wiki.org/ds/stack/>
> - *OI Wiki* 单调栈：<https://oi-wiki.org/ds/monotonous-stack/>

### 实现：顺序栈

```cpp
struct Stack{
    int values[MAXSIZE];                // 存储数据
    int top;                            // 栈顶指针
    void Stack() { top = 0; }
    void push_back(int val) { values[++top] = val; }
    int pop_back() { return values[top--] = val; }
};
```

-   顺序栈：数组 + 栈顶指针 + 栈底指针
    -   实现、使用细节
        -   入栈：插入数据、移动栈顶指针
        -   出栈：移动栈顶指针即可（旧数据无须覆盖）
    -   操作、时间效率
        -   插入、删除：$O(1)$

```cpp
typedef int IntStack[MAXSIZE+1];        // 直接在栈底记录栈顶
IntStack st;
*st = 0;                                // 初始化栈
st[++*st];                              // 入栈
st[*st--];                              // 出栈
```

##  *Queue*

![alg_queue_operation](imgs/alg_queue_operation.svg)

-   *Queue* 队列、*FIFO* 表：限定在队尾（表尾）进行插入、删除的受限线性表
    -   队列按照 *FIFO* 先进后出原则运转，被限制在仅能在队尾插入、队首删除数据
        -   入队：队尾插入数据
        -   出队：队首删除数据

-   双端队列：可同时在队首、队尾进行插入、删除操作的队列
    -   相当于栈 + 队列
        -   `push_back`：队尾插入数据
        -   `push_front`：队首插入数据
        -   `pop_back`：队尾删除数据
        -   `pop_front`：队首删除数据

-   单调（双端）队列：满足单调性的双端队列结构
    -   为保证队列单调性
        -   `push_back`：弹出所有在插入新元素后破坏队列单调性元素后，在队尾插入数据
        -   `push_front`：弹出所有在插入新元素后破坏队列单调性元素后，在队首插入数据
        -   `pop_back`：无区别
        -   `pop_front`：无区别
    -   核心同单调栈，但支持通过出队修改端点范围
        -   实操中常限制一侧仅出队，表示单方向滑动窗口
        -   可用于维护滑动窗口极值
            -   单增队列：滑动窗口极小值
            -   单减队列：滑动窗口极大值

> - *OI Wiki* 队列：<https://oi-wiki.org/ds/queue/>
> - *OI Wiki* 单调队列：<https://oi-wiki.org/ds/monotonous-queue/>

### 实现：顺序队列

```cpp
struct Queue{
    int values[MAXSIZE];                        // 存储数据
    int head, tail;                             // 队列头、尾指针
    void Queue(){ head = tail = 0; }            // 初始化队列
    void push_back(){ values[(++tail) %= MAXSIZE]; }            // 入队
    int pop_front(){ return values[(++head) %= MAXSIZE]; }      // 出队
}
```

-   顺序队列：数组 + 队首指针 + 队尾指针
    -   操作、时间效率
        -   插入、删除：$O(1)$
    -   实现、使用细节
        -   入队：插入数据、移动队尾指针
        -   出队：移动队首指针（旧数据无需覆盖）
        -   循环队列：将数组尾、首相连，即以循环方式组织队列，避免数组空间浪费

-   双栈（栈底相连）模拟队列
    -   两栈分别作为队尾栈、队首栈
        -   队首栈空时，队尾栈元素依次弹出、压入队首栈即可
    -   操作、时间效率
        -   插入、删除：可证明每个元素只被转移一次，均摊复杂度为 $O(1)$

> - *OI Wiki* 队列：<https://oi-wiki.org/ds/queue/>

##  分块表

-   分块：适当划分数据、并在划分后块上预处理部分信息，从而提高时间效率
    -   分块大小：决定时间复杂度（当然，取决于具体问题）
        -   一般设置为 $\sqrt N$：按均值不等式可证明的最优块长
        -   实务中，直接按最大数据量确定块长（保证敌手下界即可）
    -   预处理、维护信息：通用性更好，可维护很多其他数据结构无法维护信息
        -   可为各块整体维护信息：分块预处理核心
        -   支持按位置修改数据：分块不改变元素顺序（若需暂时改变顺序，则可复制至新数组中操作）
        -   支持按块懒更新：避免不必要的大规模计算、修改数据

> - *OI Wiki* 分块思想：<https://oi-wiki.org/ds/decompose/>

### 实现：块状数组

```cpp
struct BlockArray{
    int data[MAXSIZE], belong[MAXSIZE];         // 数据、数据所属块（1 起始）
    int st[MAXBLOCK], ed[MAXBLOCK];             // 块起始、结尾位置（1 起始）
    int size, szs[MAXBLOCK];                    // 数据数量、块长度（也即块数量）
    int deltas[MAXBLOCK], sums[MAXBLOCK];       // 各块信息

    void init(){
        int num = sqrt(size);                   // 块长度为 $\sqrt n$ 时较优
        for (int i=1; i<=num; i++) st[i] = size / num * (i-1) + 1, ed[i] = size / num * i;
        end[num] = size;
        for(int i=1; i<=num; i++){
            for (int j=st[i]; j<=ed[i]; j++) belong[j] = i;
            szs[i] = ed[i] - st[i] + 1;
        }
    }
}
```

-   块状数组：存储全部数据数组 + 各块起始、结尾、大小 + 各元素所属块 + 各块（整体）信息
    -   操作、时间效率：与块大小 $\sqrt n$ 相关
    -   实现、使用细节
        -   修改区间数据：块部分直接更新、块整体则懒更新
        -   问询区间数据：块部分直接计算、块整体则利用预处理信息（懒更新信息）

> - *OI Wiki* 块状数组：<https://oi-wiki.org/ds/block-array/>

### 实现：块状链表

![alg_block_linked_list](imgs/alg_block_linked_list.png)

```cpp
struct LinkedBlock{
    int size, data[BLOCKSIZE];                  // 节点数据数量、数据
    LinkedBlock * next;

    LinkedBlock() { size=0, next=nullptr, memset(data, 0, sizeof(d)); }
    void push_back(int val){ d[size++] = val, split(); }
    void split(){                               // 节点超限时分裂，保证性能
        if(size < 2 * SQRTMAXSIZE) return;
        LinkedBlock * lb = new LinkedBlock();
        copy(size+SQRTMAXSIZE, lb->data, size - SQRTMAXSIZE), size-=SQRTMAXSIZE;
    }
}
```

-   块状链表节点：存储各块数据数组 + 块（整体）信息 + 下个块指针
    -   操作、时间效率：与块大小 $\sqrt n$ 相关
    -   实现、使用细节
        -   修改区间数据：块部分直接更新、块整体则懒更新
        -   问询区间数据：块部分直接计算、块整体则利用预处理信息（懒更新信息）
        -   插入：只需要移动数据插入的块
        -   分裂：节点大小超限时分裂为两块

> - *OI Wiki* 块状数组：<https://oi-wiki.org/ds/block-list/>

#   集合数据类型

-   集合：结构中数据元素只有同属一个集合的关系

##  并查集

-   并查集：处理不交集的合并、查询问题
    -   核心操作
        -   查询：确定元素所属的子集
        -   合并：将两子集合并

### 实现：顺序并查集

```cpp
struct UFSet{
    int ufs[MAXSIZE];                                               // 存储父索引（各元素即自身）

    void init(){ for(int i=0; i<MAXSIZE; i++) ufs[i]=i; }           // 初始化并查集
    int find(int key) { return key == ufs[key] ? key : find(ufs[key]) }
    void union(int x, int y) { ufs[find(x)] = find(y); }
}
```

-   顺序并查集：数组 + 父节点指针构建树
    -   操作、时间效率
        -   查找：启用路径压缩、启发式合并后 $O(\alpha(n))$（$\alpha$ 为 *Ackermann* 函数反函数）
        -   合并：同查找
    -   实现、应用细节
        -   初始化：数组中元素值置为对应索引，即各节点属于自身集合（单节点树）
        -   祖先（树根）：值等于索引即为祖先（树根）
        -   合并：将任一子集祖先设为另外祖先儿子

> - *OI Wiki* 并查集：<https://oi-wiki.org/ds/dsu/>
> - *OI Wiki* 并查集复杂度：<https://oi-wiki.org/ds/dsu-complexity/>
> - *OI Wiki* 并查集应用：<https://oi-wiki.org/topic/dsu-app/>

####    优化：路径压缩

-   路径压缩：集合中每个节点直接指向祖先（所有节点直接指向树根）
    -   只有节点与祖先关系重要，寻找父亲路径可被压缩
    -   路径压缩一般在查找阶段顺便完成，可能造成大量修改
    -   仅应用路径压缩时，最坏 $O(m log n)$、平均 $O(m \alpha(m,n))$

    ```cpp
    int find(int x){
        Stack st;
        while(x != ufs[x]) st.push(x), x=ufs[x];    // 带路径压缩的查找
        while(st) ufs[st.pop()] = x;
        return x;
    }
    ```

####    优化：启发式合并

-   启发式合并：设计基于秩、树深度估价确定合并方向
    -   估价函数一般仅使用秩、树深度之一，时间效率均为 $O(m log n)$
    -   同时应用路径压缩、启发式合并 $O(m \alpha(m,n))$

    ```cpp
    int size[MAXSIZE] = {1};            // 需额外维护节点规模
    void union(int x, int y){
        int xp = find(x), yp = find(y);
        if (xp == yp) return;
        if (size[xp] > size[yp]) swap(xp, yp);
        ufs[xp] = yp, size[yp] += size[xp];
    }
    ```

##  哈希表（散列表）

-   哈希表：以键值对形式存储数据
    -   哈希映射（函数）：将键映射至指定位置
        -   哈希核心逻辑
        -   即，定义域为键取值范围、值域为指定整数范围（地址）的映射
    -   核心操作
        -   插入：哈希映射获取位置，在该位置上插入值
        -   查找：哈希映射获取位置，查询位置上值

-   （哈希）冲突：不同键值的哈希映射结果相同
    -   冲突不可避免（键取值数量总是大于地址取值数量），需要设计方法处理冲突
        -   *Open (Address) Hashing* 闭散列、开放寻址：哈希表直接存储所有键，冲突时按规律搜索哈希表
        -   *Chaining Hashing* 开散列、拉链：哈希表存储指向哈希桶的指针的目录项，桶中存储哈希键
    -   但，需要设计合适的哈希映射以减少冲突

-   负载因子 $\alpha = \frac {n_{set}} {n}$：哈希表的负载情况
    -   负载因子一定程度上反映
        -   查找效率
        -   冲突可能性
        -   空间利用率
    -   需将负载因子控制在合适水平以保证性能、耗费的平衡

### 实现：静态链表开散列

```cpp
struct HashTable{
    struct StaticLinkedNode{                            // 静态链表整体作为哈希桶存储数据
        int key, value, next;
    } data[MAXSIZE];
    int head[MAXSIZE];                                  // 哈希表存储目录项
    int size=0;                                         // 指示链表可用位置

    int f(int key) { return key % MAXSIZE; }            // 哈希函数
    int get(int key){
        for (int p = head[f(key)]; data[p].key != key && data[p].next; p=data[p].next);
        if(data[p].key == key) return data[p].value;
        return -1;
    }
    int set(int key, int value){
        for (int p = head[f(key)]; data[p].key != key && data[p].next; p=data[p].next);
        if(data[p].key == key) data[p].value = value;
        else data[data[p].next = ++size] = (StaticLinkedNode){key, value, NULL};    // 新元素插入桶尾
    }
}
```

-   静态链表开散列：使用静态链表作为开散列哈希桶存储数据
    -   操作、时间效率
        -   查找、赋值：$O(1)$
    -   实现、应用细节
        -   额外指针：记录静态链表表尾位置，指示可插入新键值对位置
        -   查找：从哈希表查找哈希桶指针，即桶链在静态链表中起始索引，之后遍历桶链
        -   赋值：遍历对应哈希桶链，键不存在则插入额外指针指示位置并与哈希桶串联

```cpp
struct hash_map{                            // 优化实现
    struct Node{
        long long key;
        int val, next;
    } data[MAXSIZE];
    int head[MAXSIZE], size;
    int hash_map() { size=0; memset(head, 0, sizeof(head)); }                   // 键 0 作为桶尾标记
    int f(long long key){ return key % MAXSIZE; }
    int& operator[](long long key){
        for (int p = head[f(key)]; data[p] != key && data[p].next; p=data[p].next);
        if(data[p].key == key) return data[p].val;
        data[++size] = (Node){key, -1, head[f(key)]}, head[key]=size;           // 新元素插入桶头
        return data[size].val;
    }
}
```

### 实现：数组闭散列

```cpp
struct hash_map{
    int keys[MAXSIZE];
    int vals[MAXSIZE];
    int hash_map() { memset(vals, 0, sizeof(vals)); }
    int f(long long key){ return key % MAXSIZE; }
    int& operator[](long long key){
        int p = f(key), i = 0;
        for(p, i; keys[p+i*i] != key && vals[p+i*i] != 0; i++);                 // 值 0 表示搜索终止
        if(keys[p] == key) return vals[p];
        return keys[p] = key, vals[p] = -1;
    }
}
```

-   数组闭散列：数组实现闭散列
    -   操作、时间效率
        -   查找、赋值：$O(1)$
    -   实现、应用细节
        -   两个数组分别记录键、值
        -   查找：冲突时按平方数跳跃搜索键

##  堆、优先队列

-   堆、优先队列：可按优先级依次从中获取元素的集合
    -   核心操作
        -   插入：向堆中插入新数据
        -   查询最值：获取堆中最值
        -   删除最值：删除队中最值
        -   合并：合并两个堆
        -   修改：修改队中元素的值

-   堆一般均通过树实现，但是考虑逻辑结构，将其放在集合类中
    -   优先级（权值）低至高、高至低的堆分别称为小根堆、大根堆
    -   同样的，核心是确保树高度可接受
        -   二叉堆：完全二叉树
        -   左偏树堆：维护节点距外节点距离 `dist` 保证左偏
        -   随机合并堆：合并时，随机交换左、右子树
        -   配对堆：删除树根时，配对合并、右至左依次合并兄弟链

|堆时间效率|配对堆|二叉堆|左偏树|二项堆|斐波那契堆|
|-----|-----|-----|-----|-----|-----|
|插入       |$O(1)$|$O(log n)$|$O(log n)$|$O(1)$|$O(1)$|
|查询最值   |$O(1)$|$O(1)$|$O(1)$|$O(log n)$|$O(1)$|
|删除最值   |$O(log n)$|$O(log n)$|$O(log n)$|$O(log n)$|$O(log n)$|
|合并       |$O(1)$|$O(n)$|$O(log n)$|$O(log n)$|$O(1)$|
|修改元素值 |$o(log n)$|$O(log n)$|$O(log n)$|$O(log n)$|$O(1)$|
|是否支持持久化|$O(1)$|$O(log n)$|$O(log n)$|$O(1)$|$O(1)$|

### 实现：数组二叉堆

![alg_binary_heap_array_storage](imgs/alg_binary_heap_array_storage.svg)

```cpp
struct BiHeap{
    int data[MAXSIZE], size;

    // `up`、`down` 均只调整至当前节点局部满足堆性质
    void up(int x){ while(x > 1 && data[x] > data[x / 2]) swap(data[x], data[x/2]), x /= 2; }
    void down(int x){
        while(x*2 <= size){
            int t = x * 2;
            if (t + 1 <= size && data[t] < data[t+1]) t++;          // 找到最大子节点
            if (data[t] <= data[x]) break;                          // 满足堆性质则停止
            swap(data[x], data[t]), x = t;
        }
    }
    void build_up(){ for(int i = 1; i <= size; i++) up(i); }        // O(n)
    void build_down(){ for(int i = size; i >= 1; i--) down(i); }    // O(nlogn)
    int del(){ return swap(data[1], data[size--]), down(1), data[size+1]; };
}
```

-   （大根）二叉堆：数组 + 完全二叉树 + 堆性质：父亲节点权值不小于子节点权值
    -   二叉堆依赖完全二叉树数组实现存储结构（蕴含大量元信息）保证操作复杂度
        -   完全二叉树将节点按层次遍历顺序（根节点索引置 1）放入数组中
        -   考虑到完全二叉树性质，父子节点索引满足
            -   $idx_l = 2 * idx_p$
            -   $idx_r = 2 * idx_p + 1$
    -   实现、操作细节
        -   向上：将节点与父节点交换，直到节点局部（节点、父节点）局部满足堆性质
        -   向下：将节点与子节点交换，直到节点局部（节点、子节点）局部满足堆性质
            -   向上、向下仅保证局部性质，不保证根至叶子路径上节点均满足
        -   建堆：遍历元素，通过向上、向下操作保证堆性质
            -   为保证局部可推广到全局，则 **已完成部分整体应满足堆性质**
            -   即，或顶部开始向下 $O(n log n)$
            -   或，底部开始向上 $O(n)$：堆性质较弱、结果不唯一，故可 $O(n)$ 建堆
        -   插入：在最下层、最右侧叶子节点上插入新元素，之后向上调整（交换）节点直到满足堆性质
        -   删除（最值）：交换根节点与最下层、最右侧叶子节点，之后向下调整（交换）节点直到满足堆性质
        -   修改权值：增加权值向上调整、减少权值向下调整

### 衍生：对顶二叉堆

```cpp
struct TopKBiHeap{
    BiHeap bh, sh;
    int K;

    // 初始所有元素在大根堆中
    // 删除元素，直接放入小根堆相应位置
    void build_from_bh(){ for( bh.build_down(); sh.size <= K; sh[K - (sh.size++)] = bh.del()); }
    void build_from_sh(){ for( sh.build_down(); sh.size > K; bh[sh.size - K] = sh.del()); }
    int del(){
        int val = sh.data[1];
        sh.data[1] = bh.del();              // 删除大根堆顶，直接插入小根堆顶
        return val;
    }
    void set_K(int newk){
        if (K > newk) copy(bh.data+1, bh.data+1+K-newk, bh.size), build_from_sh();
        if (K < newk) copy(sh.data+1, sh.data+1+newk-K, sh.size), build_from_bh();
    }
}
```

-   对顶堆：大根堆 + 小根堆 + 树根相对
    -   对顶堆可用于实现可变第 $K$ 大元素的获取删除
    -   实现、操作细节
        -   维护：不断取出大根堆、小根堆树根插入对方堆中，直到小根堆元素数量为 $K$
        -   插入：若插入元素大于小根堆树根，插入小根堆，否则插入大根堆，维护对顶堆
        -   删除：删除小根堆树根，维护对顶堆
        -   $K$ 值修改：根据 $K$ 值维对顶堆

> - *OI Wiki* 二叉堆：<https://oi-wiki.org/ds/binary-heap/>

### 实现：左偏树堆

```cpp
typedef struct{
    int val, dist;
    Node * l, * r;
} Node;
Node * merge(Node * x, Node * y){
    if (x == nullptr) return y;
    if (y == nullptr) return x;
    if (x->val < y->val) swap(x, y);
    x->r = merge(x->r, y);                              // 递归合并右子树、较小树
    if(x->l->dist < x->r->dist) swap(x->l, x->r);       // 交换左、右子树保证左偏性质
    return x->dist = x->r->dist + 1, x;
}
Node * del(Node * x){ return merge(x -> l, x -> r); }
```

-   （大根）左偏树堆：二叉树 + 左偏 + 堆性质：父亲节点权值不小于子节点权值
    -   左偏树堆通过维护节点到外节点（存在空儿子的节点） *dist* 确定合并顺序
        -   左偏树定义即 `l->dist > r->dist`，则总应有 `fa->dist = r->dist + 1`
        -   左偏树不保证深度
    -   实现、细节、效率
        -   合并：较大根作为新根，递归合并右子树、较小根树，若需则交换左、右节点保证左偏
        -   插入：单节点视为左偏树与原堆合并
        -   删除：合并左右子树

####    优化：更多操作支持

```cpp
typedef struct{
    int val, dist;
    Node * l, * r, * fa;                        // 增加指向父节点指针
} Node;
void up_update_dist(Node * x){                  // 向上更新 `dist` 直至根节点
    if (x == nullptr) return;                   // 更新 `dist` 至根节点
    x->dist = x->r->dist + 1, up_update_dist(x->fa);
}
Node * merge(Node * x, Node * y){
    if ( x == nullptr ) return y;
    if ( y == nullptr ) return x;
    if ( x->val < y->val ) swap(x, y);
    x->r = merge(x->r, y), x->r->fa = x;
    return up_update_dist(x), x;
}
void del(Node * x){ up_update_dist(merge(x->l, x->r)->fa = x->fa); }
```

-   左偏树堆增加父节点指针后，可实现更多操作
    -   删除任意节点：合并节点左、右子树，替代原节点，向上更新 *dist*

### 随机合并（二叉）堆

```cpp
typedef struct{
    int val;
    Node * l, * r;
} Node;
Node * merge(Node * x, Node * y){
    if (x == nullptr) return y;
    if (y == nullptr) return x;
    if (x->val < y->val) swap(x, y);
    if (rand() & 1) swap(x->l, x->r);                   // 随机交换左、右子树
    x->r = merge(x->r, y);                              // 递归合并右子树、较小根树
    return x;
}
```

-   （大根）随机合并堆：二叉树 + 合并时随机交换左右子树 + 堆性质：父亲节点权值不小于子节点权值

> - 随机堆效率证明：<https://cp-algorithms.com/data_structures/randomized_heap.html>

### 实现：多叉树配对堆

![alg_pairing_heap_structure](imgs/alg_pairing_heap_structure.png)
![alg_pairing_heap_storage](imgs/alg_pairing_heap_storage.png)

```cpp
typedef struct{
    int w;
    Node * child, * sibling;            // 儿子、兄弟表示法存储多叉树
} Node;
Node * meld(Node * x, Node * Y){        // 合并两个堆
    if (x == nullptr) return y;
    if (y == nullptr) return x;
    if (x->w < y->w) swap(x, y);
    return y->sibling = x->child, x->child = y, x;
}
Node * merge(Node * x){                                     // 核心、辅助：合并兄弟链
    if (x == nullptr || x->siblings == nullptr) return x;
    Node * y = x->sibling, * c = y->sibling;
    x->sibling = y->sibling = nullptr;
    return meld(merges(c), meld(x, y));                     // 两两合并 -> 从右往左依次合并
}
Node * del(Node * root){ return merge(root->child); }
```

![alg_pairing_heap_del_step](imgs/alg_pairing_heap_del_step_1.png)
![alg_pairing_heap_del_step](imgs/alg_pairing_heap_del_step_2.png)


-   （大根）配对堆：多叉树 + 堆性质：父亲节点权值不小于子节点权值
    -   配对堆完全不维护树的元数据：大小、深度、排名
        -   任何满足堆性质的树都是合法的配对堆，在实践中效率优秀
        -   但需，通过精心设计的操作顺序保证均摊复杂度
    -   实现、细节
        -   合并：较大根节点作为新根节点，较小根节点作为最左侧儿子
        -   插入：新元素视为新配对堆与原堆合并
        -   删除（最大值）：两两合并儿子（即配对）、按从右往左顺序依次合并
            -   删除操作被精心设计顺序保证以保证均摊复杂度
        -   增加节点值：剖出子树与原树合并
            -   需修改节点结构，添加指向父节点、左兄弟的指针
    -   配对堆基于势能分析的时间效率分析复杂，仍未有紧结论
        -   合并、删除均摊：$O(log n)$
        -   增加节点值：$O(2^{2 * \sqrt {log log n}})$

> - *OI Wiki* 配对堆：<https://oi-wiki.org/ds/pairing-heap/>
> - *Brilliant* 配对堆教程：<https://brilliant.org/wiki/pairing-heap/>
> - *The Pairing Heap: A New Form of Self-Adjusting Heap*：<http://www.cs.cmu.edu/~sleator/papers/pairing-heaps.pdf>

#   树数据类型

-   树型结构：结构中数据元素存在一对多关系
    -   自由树：由 $n$ 节点、$n-1$ 边组成的无向连通图
    -   （有根）树：存在（指定有）根节点的自由树
        -   最平常意义的树，树不特意说明即指有根树
        -   在自由树基础上指定根节点，由此确定树的层次（父子）关系
    -   有序树：节点子节点有序的有根树
        -   最常使用的树：为保证性能，子节点往往需要根据节点权值排序
        -   在有根树的基础上固定子节点顺序，由此确定节点次序（兄弟）关系

##  有序树

### 实现：链树

```cpp
typedef struct TNodePLinked{
    int val;
    struct TNodePLinked * p;                    // 1、父节点指针
    struct TNodePLinked *fc, *sib;              // 2、首个孩子节点、下个兄弟指针
    struct TNodePLinked (*chs)[MAXCHILD];       // 3、孩子节点指针数组
} TNodePLinked;
```

-   链树（节点）：权值（数据）+ 父节点指针 + 孩子节点指针
    -   操作、时间复杂度
        -   插入：$O(1)$
        -   查找双亲：设置有父节点指针时 $O(1)$、否则 $O(h)$
        -   查找儿子：设置有孩子节点指针时 $O(1)$、否则 $O(n)$（需要遍历所有节点）
    -   实现、使用细节
        -   父节点指针：仅包含父节点指针的实现较少使用，大部分操作效率太低
        -   孩子节点指针数组：仅适合各节点子女数量变化不大场合，否则浪费空间
        -   孩子、兄弟指针：此即体现多叉树、二叉树一一对应

### 实现：静态链树

```cpp
typedef struct TNodePStatic{
    int val;
    int p;                                      // 1、父节点位置
    fc, sib;                                // 2、首个孩子节点、下个兄弟位置
    int chs[MAXCHILD];                          // 3、孩子节点位置数组
} TNodePStatic;
struct TTreePStatic{
    TNodePStatic data[MAXSIZE];
    int size;
}
```

-   静态链树：数组存储节点数据 + 数组索引表示连接关系

##  二叉树

-   二叉树：顶点子女数最多为 2 的有序树
    -   二叉树特点
        -   第 $i$ 层二叉树最多 $2^{i-1}$ 节点
        -   高度 $k$ 二叉树最多 $2^k - 1$ 节点
        -   记 $n,n_0,n_1,n_2$ 为节点总数、度 $0,1,2$ 节点数，则有
            -   $n = n_0 + n_1 + n_2 = n_1 + 2n_2 + 1$：点等式、边等式
            -   $n_0 = n_2 + 1$
    -   满二叉树：最后层节点为 0、其余层节点度为 2 二叉树
    -   完全二叉树：与满二叉树节点按顺序对应的二叉树
        -   将树节点按层次、顺序从 1 开始编号，则父子节点索引满足
            -   $idx_l = 2 * idx_p$
            -   $idx_r = 2 * idx_p + 1$
    -   二叉搜索树：满足 “左子树权值 < 节点权值 < 右子树权值” 带权二叉树

-   二叉树的计数：$n$ 个节点、互不相似的二叉树数量 $b_n$
    -   任意有序树与没有右子树的二叉树一一对应（参考儿子、兄弟双指针结构）
    -   考虑二叉树由根节点、左右子树构成，则有如下递推式、求解

        $$
        \left \{ \begin{array}{l}
        b_0 & = 1 \\
        b_n & = \sum_{i=0}^{n-1} b_i b_{n-i-1}, & n \geq 1
        \end{array} \right. \Rightarrow
        b_n = \frac 1 {n+1} \frac {(2n)!} {n!n!} = \frac 1 {n+1} C_{2n}^n
        $$

### 实现：静态完全二叉树

```cpp
typedef int CompleteBinaryTree[MAXSIZE+1];
```

-   静态完全二叉树：数组存储数据 + 完全二叉树父子节点关系
    -   操作、时间复杂度
        -   查找父、子节点：$O(1)$，借助父、子节点索引关系直接定位
        -   插入：$O(n)$
    -   实现、使用细节
        -   数组 0 位置：存储节点数目，则无需额外变量维护数节点数目
        -   一般二叉树可按对应的完全二叉树静态存储，但浪费空间

### 实现：链二叉树

```cpp
struct BTNode{
    int val, size;                                      // 权、规模
    struct BTNode *lc, *rc;
}
void preorder(BTNode * root){                           // 先序遍历
    std::stack<int> s;
    BTNode * cur = root;
    while(!s.empty() || cur){
        while(cur){
            VISIT(cur);
            s.push(cur), cur = cur->lc;
        }
        cur = s.pop(), cur = cur->rc;
    }
}
void inorder(BTNode * root){
    std::stack<int> s;
    BTNode * cur = root;
    while(!s.empty() || cur){
        while (cur) s.push(cur), cur = cur->lc;
        cur = s.pop(), VISIT(cur), cur = cur->rc;
    }
}
void postorder(BTNode * root){
    std::stack<int> s;
    BTNode * cur = root, last;
    while(!s.empty() || cur){
        while(cur) s.push(cur), cur = cur->lc;
        cur = s.top();
        if(cur->rc == nullptr || cur->rc == last){      // 根据右儿子是否访问过，判断压栈、出栈
            VISIT(cur);
            last = s.pop(), cur = nullptr;
        }else cur = cur->rc;
    }
}
```

-   链二叉树节点：数据 + 左、右儿子指针
    -   实现、使用细节
        -   先序遍历：访问、压栈、左儿子、出栈、右儿子、压栈
        -   中序遍历：压栈、左儿子、出栈、访问、右儿子、压栈
        -   后序遍历：压栈、左儿子、（右儿子、压栈）或（出栈、访问）
            -   出栈前判断右儿子是否已访问：维护上次访问节点，与右儿子比较

### 实现：二叉搜索树

```cpp
int getmin(BTNode * root){ return root->lc ? root->val : getmin(root->lc); }
void insert(BTNode *& root, int val){               // 指针的引用类型作为参数，无需维护父节点以修改其指针
    if (root == nullptr) root = new BTNode(val, 1, 1, nullptr, nullptr);
    root->size++;                                   // 下推时即修改树规模
    else if (root->val < val) insert(root->lc, val);
    else insert(root->rc, val);                     // 递归插入
}
void del_val_node(BTNode *& root, int val){             // 交换删除：与右子树最小值交换后删除
    root->size--;                                       // 下推时即可修改规模
    if (root->val == val){
        if(root->lc == nullptr || root->rc == nullptr)  // 单分支时可直接删除
            root = root->lc | root->rc;                 // 此处节点空间未释放
        else{
            BTNode * tmp = root->rc;
            while(tmp->lc != nullptr) tmp = tmp->lc;    // 查找右子树最小值
            std::swap(root->val, tmp->val);             // 交换节点信息、维持节点指针
            del_val_node(root->rc, val);                // 递归在右子树删除原值，最小值节点单分支
        }
    }else if (root->val > val) del_val_node(root->lc, val);
    else del_val_node(root->rc, val);
}
BTNode * merge(BTNode * x, BTNode * y){                 // 归并两颗树
    if(x == nullptr || y == nullptr) return x | y;
    if(x->val > y->val) std::(x, y);
    BTNode * tmp = y->lc;
    x->rc = merge(x->rc, y->lc), y->lc = x;             // 较大节点做根
    return y;
}
void del_val_node_merge(BTNode *& root, int val){       // 归并删除：合并子树替代原树根
    root->size--;
    if(root->val == val) root = merge(root->lc, root->rc);  // 此处节点空间未释放
    else if (root->val > val) del_val_node_merge(root->lc, val);
    else del_val_node_merge(root->rc, val);
}
```

-   二叉搜索树：二叉树 + 左、右子树权值有序
    -   权值有序即中序遍历有序
    -   实现、使用细节
        -   查找最小、最大：不断查找左、右儿子
        -   插入：与节点比较以确定插入左、右子树
        -   交换删除：节点清空时使用右子树最小值（或左子树最大值）替换被删除节点
            -   可尽量维持树形状不变
        -   归并删除：使用左、右儿子替代被删除节点，向下合并二叉树

> - *OI Wiki* 二叉搜索树：<https://oi-wiki.org/ds/bst/>

####    优化：最优二叉搜索树
#TODO

##  平衡（自）二叉（搜索）树

```cpp
typedef struct BSTNode{
    int val, w;                             // 局部（节点、边）性质：值、权
    int size, h;                            // 子树性质：规模、高度
    struct BTNode * lc, * rc, * fa;
}BSTNode;
void update_info(BSTNode * root){
    if (root->lc == root->rc == nullptr) root->h = 1, root->size = 1;
    elseif (root->lc == nullptr) root->h = root->rc->h + 1, root->size = root->rc->size + 1;
    elseif (root->rc == nullptr) root->h = root->lc->h + 1, root->size = root->lc->size + 1;
    else root->h = std::max(root->lc->h, root->rc->h) + 1, root->size = root->lc->size + root->rc->size;
}
```

-   *self-Balancing binary Search Tree*：在任意插入、删除情况下保证低高度的二叉搜索树
    -   自平衡（降低高度）通过在插入、删除、查找操作后执行变换保证，基础操作一般即旋转
    -   旋转本质即保证权值有序的情况下，上升、下降边两侧节点位置（交换父子关系）
        -   即，旋转将交换、调整子树位置，导致树高度变化（列式即可看出）
        -   故，需根据树情况选择合适的旋转操作

```cpp
BSTNode * zig(BSTNode * root){                  // 右旋提升左子树，返回新根
    BSTNode * tmp = root->lc;
    root->lc = tmp->rc, tmp->rc = root;         // 调整儿女指针
    tmp->fa = root->fa, root->fa = tmp;         // 调整父指针
    if(root->lc != nullptr) root->lc->fa = root;// 需判断节点是否为空
    update_info(root), update_info(tmp);        // 由下至上两节点信息，上层节点由调用者负责
    return tmp;
}
BSTNode * zag(BSTNode * root){                  // 左旋提升右子树，返回新根
    BSTNode * tmp = root->rc;
    root->rc = tmp->lc, tmp->lc = root;
    if(root->rc != nullptr) root->rc->fa = root;
    tmp->fa = root->fa, root->fa = tmp;
    update_info(root), update_info(tmp);        // 由下至上更新两节点信息，上层节点由调用者负责
    return tmp;
}
```

-   （单）旋转即交换树中一条边的父子关系
    -   从父节点角度：儿女节点不确定，可分为右（上）旋转 *zig*、左（上）旋转 *zag*
        -   右旋 *zig*：交换父节点、左儿子父子关系
            -   左孩子右上旋作根
            -   节点作左孩子右儿子
            -   左孩子的右孩子作节点左孩子
        -   左旋 *zag*：交换父节点、右儿子父子关系
            -   右孩子左上旋作根
            -   节点作有孩子左儿子
            -   右孩子的左孩子作节点右孩子
    -   从子节点角度：父节点确定，旋转方式是确定的
    -   旋转本身会改变树结构，若树中节点中信息依赖子树，则须沿路径向上更新至根节点，实现中：
        -   旋转动作一般只负责更新旋转边两侧节点信息
        -   **通过自底向上的调用旋转以维护路径中可能受影响节点信息**

```cpp
BSTNode * up(BSTNode * cur){                                // 从子节点角度，`zig`、`zag` 是可合并的
    if(cur->fa == nullptr) return nullptr;
    BSTNode * fa = cur->fa;
    BSTNode *& curp = fa->lc == cur ? cur->rc : cur->lc;    // 获取需修改的指针的引用
    BSTNode *& fap = fa->lc == cur ? fa->lc : fa->rc;
    fap = curp, curp = fa;                                  // 修改节点指针信息，同 `zig`、`zag`
    if(fap != nullptr) fap->fa = fa;
    tmp->fa = fa->fa, fa->fa = tmp;
    update_info(fa), update_info(cur);
    return cur;
}
```

> - *OI Wiki* 平衡树：<https://oi-wiki.org/ds/bst/#平衡树简介>
> - *Wiki BST*：<https://en.wikipedia.org/wiki/Self-balancing_binary_search_tree>
> - *Wiki* 树旋转：<https://en.wikipedia.org/wiki/Tree_rotation>

### 实现：*AVL* 树

![alg_avl_tree_rotation_example](imgs/alg_avl_tree_rotation_example.png)

-   *AVL* 树：二叉搜索树 + 维护节点高度 + 4 种旋转 + 回退时（自底向上）旋转保证平衡因子
    -   *AVL* 树思想：通过旋转保证平衡因子（左、右子树高度差）不超过 1，保证查找效率
        -   在已平衡的 *AVL* 树中插入、删除单个节点对树高影响不超过 1
        -   平衡因子超界限时，沿修改节点向上维护即可恢复平衡
            -   自底向上维护保证下层子树已经恢复平衡
        -   可证明，*AVL* 树高度（查找效率、复杂度） $\in \Theta(log n)$
    -   实现、使用细节
        -   旋转：节点、子节点高侧相同，节点反向旋转即可；高侧不同则子节点反向旋转至高侧相同
        -   插入：同普通二叉搜索树，回退时旋转节点（维护信息）即可
        -   交换删除：同普通二叉搜索树，回退时旋转节点（维护信息）即可
    -   *AVL* 树特点
        -   需额外维护高度信息维持平衡
        -   插入、删除后需旋转代价较高

```cpp
int inline geth(BSTNode * r) { return r == nullptr ? r->h : 0; }
void rotate(BSTNode * root){                                    // 检查平衡因子、旋转
    int lh = geth(root->lc), rh = geth(root->rc);
    if(lh - rh < 2 and rh - lh < 2) return;                     // 无需旋转
    else if(lh > rh){                                           // 判断节点高侧
        int llh = geth(root->lc->lc), lrh = geth(root->lc->rc);
        if(llh < lrh) zag(root->lc);                            // 修正为高侧相同
        zig(root);                                              // 最后的单旋
    }else{
        int rlh = geth(root->rc->lc), rrh = geth(root->rc->rc);
        if(rrh < rlh) zig(root);
        zag(root);
    }
}
```

> - *OI Wiki AVL* 树：<https://oi-wiki.org/ds/avl/>
> - *AVL* 树介绍：<https://www.cnblogs.com/ZhaoxiCheung/p/6012783.html>，此文章中删除节点部分仅在右子树空情况在终止，因此在某些情况会执行多次不必要结构改变（节点交换）
> - *AVL* 树可视化：<https://www.cs.usfca.edu/~galles/visualization/AVLtree.html>

### 实现：*Splay* 树

![alg_splay_tree_2rotation_splay](imgs/alg_splay_tree_2rotation_splay.png)

-   *Splay* 树、伸展树：二叉搜索树 + 6 种旋转 + （自底向上）旋转直至节点为根
    -   *Splay* 树思想：通过精心设计的伸展动作将节点旋转至根，降低（查找）路径高度、修复树结构
        -   伸展动作类似 *AVL* 树，旋转考虑祖、父、子三节点，但逻辑不同
            -   *AVL*（平衡状态下）在插入、删除节点后旋转以恢复局部平衡
        -   *Splay* 中 4 种双旋设计则在推高子节点 2 次，同时降低路径高度
            -   之式：连续两次上旋子节点（同 *AVL* 树）
            -   链式：先上旋父节点、再上旋子节点（若两次均上旋子节点，则无法压缩路径高度）
        -   *Splay* 中 2 种单旋仅用于收尾场合（路径仅剩一边），将节点旋转至根
    -   实现、使用细节
        -   自底向上伸展：若维护有父指针，可沿父指针向上伸展至根；否则回退时沿压栈路径伸展
        -   合并：要求两棵树整体有序，将较小树最大值伸展至根、右指针连接至较大树根
        -   插入：同普通二叉搜索树，之后伸展插入节点
        -   删除：将待删除节点伸展至树根，之后合并两棵子树
    -   *Splay* 树特点
        -   无需额外维护信息（如高度）等维持平衡
        -   局部性强，适合数据访问满足局部性特点场合
        -   均摊复杂度 $O(log N)$，但无法避免最坏情况

```cpp
inline bool isl(BSTNode * root) { return root->fa != nullptr && root->fa->lc == root; }
BSTNode * splay_botup(BSTNode * root){              // 沿指针向上，根节点被修改为传入节点
    while(root->fa != nullptr){                     // 树关系在旋转过程中被维护（修改）
        if(root->fa->fa == nullptr) break;
        if(isl(root->fa) ^ isl(root)) up(root), up(root);       // zig-zag + zag-zig
        else up(root->fa), up(root);                            // zig-zig + zag-zag
    }
    return up(root);                                            // zig + zag
}
```

-   *Splay* 树支持自顶向下伸展：即在查找过程中分裂、挂载至左右挂载点
    -   维护左、中、右三棵树拆分、暂存查找过程中伸展结果，最后合并三棵树
        -   中树为待查找树，其树根即查找路径当前节点
        -   左、右树暂存小于中树、大于中树节点
        -   左、右树增加挂载点 + 仅祖父边旋转 + **父子边切分重连至左、右树**
            -   保证中树根节点原儿女指针不变
            -   左、右树挂载点即被切分后、插入的根节点右儿子、左儿子
            -   否则，旋转将导致中树根节点重连祖、父，重复查找
    -   与自底向上伸展压缩路径逻辑相同，结果因单旋可能有差异

![alg_splay_tree_topdown_splay](imgs/alg_splay_tree_topdown_splay.png)

```cpp
inline void merge_2node(BSTNode *& fa, BSTNode * ch){
    if(fa->val > ch->val) fa->lc = ch, ch->fa = fa;
    else fa->rc = ch, ch->fa = fa;
    fa = ch;
}
inline BSTNode * merge_lmr(BSTNode * mt, BSTNode * ltr, BSTNode * rtr,
    BSTNode * ltc, BSTNode * rtc){                  // 此函数维护自底向上信息无意义，省略
    ltc->rc = mt->lc, mt->lc->fa = ltc;
    rtc->lc = mt->rc, mt->rc->fa = rtc;
    mt->lc = ltr, ltr->fa = mt;
    mt->rc = rtr, rtr->fa = mt;
    return mt;
}
BSTNode * splay_topdown(BTNode * mt, int val){
    BSTNode * ltr, * rtr, * ltc, *rtc;              // 左树、右树、左树联结点、右树联结点
    while(mt->val != val && mt != nullptr){
        if(mt->val == val) return merge_lmr(mt, ltr, rtr, ltc, rtc);
        if(mt->val > val){
            if(mt->lc->val > val && mt->lc->lc != nullptr)      // zig-zig
                mt = zig(mt->lc), merge_2node(rtc, mt), mt = mt->lc;
            if(mt->lc->val < val && mt->lc->rc != nullptr)      // zig-zag
                merge_2node(rtc, mt), merge_2node(ltc, mt->lc), mt = mt->lc->rc;
            else return zig(mt), merge_lmr(mt, ltr, rtr, ltc, rtc);
        }else{
            if(mt->rc->val < val && mt->rc->rc != nullptr)      // zag-zag
                mt = zag(mt->rc), merge_2node(ltc, mt), mt = mt->rc;
            if(mt->rc->val > val && mt->rc->lc != nullptr)      // zag-zig
                merge_2node(ltc, mt), merge_2node(rtc, mt->rc), mt = mt->rc->lc;
            else return zag(mt), merge_lmr(mt, ltr, rtr, ltc, rtc);
        }
    }
    return mt;
}
```

> - *OI Wiki Splay* 树：<https://oi-wiki.org/ds/splay/>
> - *Splay* 树自底向上：<https://josephjsf2.github.io/data/structure/and/algorithm/2020/05/03/splay-tree.html>
> - *Splay* 树自顶向下：<https://www.jianshu.com/p/c57e6c851225>
> - *Splay* 树自顶向下：<https://www.cnblogs.com/huangxincheng/archive/2012/08/04/2623455.html>
> - *Splay* 树旋转设计：<https://zhuanlan.zhihu.com/p/348797577>

### 实现：*Treap*

-   树堆：值二叉搜索树 + 权堆（可不满）
    -   树堆思想：给节点随机赋权，操作后通过维护权堆性质时期望树结构不退化
    -   旋转树堆：通过旋转维护权堆性质
        -   旋转：检查节点、儿女权堆性质，旋转维护局部堆性质
        -   插入：同普通二叉搜索树，回退时考虑旋转
        -   删除：将待删除节点权赋 0（假设权非负），向下旋转至有女儿指针为空，之后删除节点

```cpp
void rotate(BSTNode * root){                        // 检查权（大根）堆性质、旋转维护
    int lv = root->lc == nullptr ? 0 : root->lc->val;
    int rv = root->rc == nullptr ? 0 : root->rc->val;
    if(lv > rv && lv > root->w) zig(root);
    else if(rv > lv && rv > root->w) zag(root);
}
```

-   无旋树堆：通过分裂、合并保证权堆性质
    -   分裂：沿查找路径切断边、重连边，建立两棵树堆
        -   按值分裂：**值节点位于左树堆最优、或右树堆最左**，即左、值、右有序
        -   按位次分裂：位次节点同上，左树堆规模即为位次数（需维护规模）
    -   合并：将值有序的两个树堆按权递归合并（保证堆性质）

```cpp
std::pair<BSTNode *, BSTNode *> split_by_val(BSTNode * root, int val){
    if(root == nullptr) return {nullptr, nullptr};  // 仅在节点不存在触发
    if(root->val == val){                           // 到目标节点终止，查找节点位于右树最左
        auto tmp = root->lc, root->lc = nullptr;    // 未维护父指针
        return update_info(root), {tmp, root};
    }
    else if(root->val < val){                       // 切分右子树
        auto tmp = split_by_val(root->rc, val);
        root->rc = tmp.first, update_info(root);    // 若无需维护子树信息，可类似 *Splay* 树自顶向下切分
        return {root, tmp.second};
    }else{
        auto tmp = split_by_val(root->lc, val);
        root->lc = tmp.second, update_info(root);
        return {tmp.first, root};
    }
}
inline int getlsz(BSTNode * root){ return root->lc ? root->lc->size : 0; }
std::pair<BSTNode *, BSTNode *> split_by_nth(BSTNode * root, int nth){
    if(getlsz(root) == nth-1){                      // 到目标节点终止，查找节点位于右树最左
        auto tmp = root->lc, root->lc = nullptr;    // 未维护父指针
        return update_info(root), {tmp, root};
    }
    else if(getlsz(root) < nth-1){                  // 切分右子树
        auto tmp = split_by_val(root->rc, nth - getlsz(root));
        root->rc = tmp.first, update_info(root);    // 若无需维护子树信息，可类似 *Splay* 树自顶向下切分
        return {root, tmp.second};
    }else{
        auto tmp = split_by_val(root->lc, nth - getlsz(root));
        root->lc = tmp.second, update_info(root);
        return {tmp.first, root};
    }
}
BSTNode * merge(BSTNode * lr, BSTNode * rr){
    if(lr == nullptr || rr == nullptr) return lr & rr;
    if(lr->w >= rr->w) std::swap(lr, rr);
    rr->lc = merge(lr, rr->lc), update_info(rr);
    return rr;
}
```

> - *OI Wiki Treap*：<https://oi-wiki.org/ds/treap/>
> - 无旋 *Treap*：<https://www.luogu.com.cn/blog/Chanis/fhq-treap>
> - 大部分 Demo 中按值分裂在查找到目标节点后，继续无意义分裂直至叶子节点为止，此处将在查到目标节点后直接停止

##  多路平衡查找树


### 实现：B 树

-   B 树可通过最小度定义，最小度为 $T$ 的 B 树需满足如下性质
    -   键数限制：节点中键升序排列
        -   除根节点外所有节点有最少有 $T-1$ 个键、至多 $2T-1$ 个键
        -   根节点至少包含 $1$ 个键
    -   键、子树（节点）关系：节点度为 $t$
        -   非叶子节点中，子树数量为键数加一 $t+1$
        -   第 $i$ 子树上键取值在键 $t_i, t_{i+1}$ 之间
    -   所有叶子节点在同一层

-   B 树常用于磁盘数据检索
    -   可包含多个键的大节点可减少磁盘读取次数，减少 *I/O* 时间
        -   实务中，B 树节点大小与磁盘块大小相同

```cpp
typdef struct{
    int keys[2*T-1], nk;        // 键值数组、实际键数
    BTreeNode * chs[2*T];       // 子节点指针数组（可考虑链表提高空间、插入效率）
    int leaf;                   // 叶子节点标记
    BTreeNode(int k) { keys[0] = k, leaf = 1, n = 1; }
} BTreeNode;
BTreeNode * search(BTreeNode * root, int k){
    int i = 0;
    for(i=0; k <= root->keys[i] && i < root->nk; i++);      // 可考虑二分定位
    if(root->keys[i] == k) return root;
    if(root->leaf) return nullptr;
    return search(root->chs[i], k);
}
BTreeNode * insert(BTreeNode * root, int k){
    if(root == nullptr) return new BTreeNode(k);
    int snk = 0;
    BTreeNode * cur = root, * fa = nullptr;
    while(cur){
        if(cur->nk == 2*T - 1){
            if(fa == nullptr) fa = new BTreeNode(), root = fa;
            split_child(fa, cur, snk);                      // 分裂后再考虑查找
            if(k <= fa->keys[snk]) cur = fa->chs[snk];      // 确定需查找的分裂后节点
            else cur = fa->chs[snk+1];
        }
        for(snk = 0; k <= cur->keys[snk] && snk < cur->nk; snk++);
        fa = cur, cur = cur->chs[snk];
    }
    copy(fa->keys+snk, fa->keys+snk+1, fa->nk-snk);         // 平移叶子节点键，插入
    fa->keys[snk] = k;
    return root;
}
void split_child(BTreeNode * fa, BTreeNode * lnode, int snk){
    // 分裂子节点
    BTreeNode * rnode = new BTreeNode();
    copy(lnode->keys+T, rnode->keys, T-1);                  // 复制、分裂键
    if(! lnode->leaf) copy(lnode->chs+T, rnode->chs, T);    // 复制、分裂指针
    lnode->nk = T-1, rnode->nk = T-1;                       // 修改键数

    // 修改父节点
    copy(fa->keys+snk, fa->keys+snk+1, fa->nk-snk);         // 平移键、指针，腾空
    copy(fa->chs+snk+1, fa->keys+snk+2, fa->nk-snk);
    fa->keys[snk] = lnode->keys[T], fa->nk++;               // 插入新键
    fa->chs[snk] = lnode, fa->chs[snk+1] = rnode;           // 设置新指针
}
BTreeNode * delete(BTreeNode * root, int k){
    std::stack<BTreeNode> nst;
    std::stack<int> dnkst;
    int dnk = 0;
    BTreeNode cur = root;
    while(cur != nullptr){
        for(dnk=0; k < cur->keys[dnk] && dnk < cur->nk; dnk++);
        if(cur->keys[dnk] == k) break;
        else st.push_back(cur), dnkst.push_back(dnk), cur = cur->chs[dnk];
    }
    while(!dnkst.empty()){
        if(cur->leaf){
            if (cur->nk > T) copy(cur->keys+dnk+1, cur->keys+dnk, cur->nk-dnk), cur->nk--;  // 直接删除
            else{
                BTreeNode * fa = nst.top();
                int fadnk = dnkst.top();
                if(fadnk == 0){
#TODO
                }
            }
        }
    }
}
```

> - *OI Wiki* B 树：<https://oi-wiki.org/ds/b-tree/>
> - B 树 *C++* 实现：<https://zhuanlan.zhihu.com/p/28996369>

##  预处理树

### 树状数组
#TODO

```cpp
int lowbit(int n){ return x & -x; }                         // 整数最低 bit
struct TreeArray{
    int data[maxsize], tree[MAXSIZE];

    int op(int a, int b) { return a+b; }
    int rop(int a, int b) { return a-b; }
    void init_bottom_up(){                                  // 自底向上 O(n) 建树
        memset(tree, 0, sizeof(tree));
        for (int i=1; i <= n; i++){                         // 按顺序更新自身时
            tree[i] = op(tree[i], data[i]);
            int j = i + lowbit(i);
            if (j <= n) tree[j] = op(tree[j], tree[i]);     // 更新直接父节点
        }
    }
    void init_with_prefix(){                                // 前缀 O(n) 建树
        int prefix[MAXSIZE];
        memset(prefix, 0, sizeof(prefix));
        for(int i=1; i <= *tree; i++) prefix[i] = op(prefix[i-1], data[i]);
        for(int i=1; i <= *tree; i++) tree[i] = rop(prefix[i], prefix[i-lowbit(i)+1]);
    }
    void op_update(int idx, int delta){
        while(idx <= *tree) tree[idx] = op(tree[idx], delta), idx += lowbit(idx);
    }
    void query_prefix(int idx){                             // 问询前缀结果
        int res;
        while(idx >= 1){
            res = op(res, tree[idx]);
            idx -= lowbit(idx);
        }
        return res;
    }
}
```

![alg_tree_array_structure](imgs/alg_tree_array_structure.svg)

-   树状数组：父节点管理、存储子女节点预处理信息的多叉树（森林）
    -   树状数组中节点数与数据量相同，节点与数据点一一对应，若从 1 开始编号
        -   编号 $n$ 节点管理 $[n - n_{lowbit} + 1,n]$ 间数据，即按照
        -   编号 $n$ 节点位于倒数 $log_2 n_{lowbit}$ 层
        -   编号 $n$ 节点被 $n_{zeros} + 1$ 个节点管理
    -   实现、使用细节
        -   自底向上建树：更新节点自身结果时，同时更新直接父节点结果
        -   前缀结果建树：构建前缀结果，通过前缀结果逆运算求节点结果

### *ST* 表

![alg_st_query_template](imgs/alg_st_query_template.svg)

-   可重复贡献问题：若运算 $opt$ 满足 $x opt x = x$，则对应的区间问询即为可重复贡献问题
    -   即，区间问询中元素可被重复计算而不影响结果，如
        -   *RMQ* 问题中最大值运算 $max(x, x) = x$
        -   区间 *GCD* 问题中最小公倍数运算 $gcd(x, x) = x$
    -   另外，运算 $opt$ 还应满足结合律

![alg_st_query_example](imgs/alg_st_query_example.svg)

-   *ST* 表：基于倍增思想，用于解决 **可重复贡献问题** 的数据结构
    -   *ST* 表可视为是在父节点存储子节点运算结果的半截二叉树
        -   树的边是交叉的：不支持修改操作
        -   只有 $\frac {log n} 2$ 高度，缺少树根部分
    -   预处理建立 *ST* 表 $\in \Theta(n log n)$
        -   记 $f(i,j) = opt([i, i+2^j-1])$，即区间 $[i,i+2^j-1]$ 的操作结果
            （闭区间，元素数量为 $2^j$）
        -   *ST* 表状态转移方程 $f(i, j) = opt(f(i,j-1), f(i+2^{j-1}, j-1))$
    -   区间问询 $\in \Theta(1)$
        -   对区间 $[l,r]$，分成两部分 $[l,l+2^s-1]$、$[r-2^s+1,r]$ 考虑
        -   考虑到问题可重复贡献，所以对任意 $[l,r]$，仅需 $s = \lfloor log_2 (r-l+1) \rfloor$ 保证覆盖、不越界即可

```cpp
int LOGT[MAXSIZE+1];
int init_logt(){                                    // 初始化对数表，避免对数计算效率低
    LOGT[1] = 0;
    for(int i=1; i<LOGSIZE; i++) LOGT[i] = LOGT[i/2] + 1;
}
class SparseTable{
    int st[MAXSIZE+1][MAXLOG];                      // 运算性质，原始数据可以直接存储在首行
    int size;

    int op(int l, int r){ return max(l,r); }        // 可重复贡献运算
    void init_st(){
        for(int j=1; j<LOGT[size]; j++){            // 递推构建 ST
            int pr = (1 << (j-1));
            for(int i=0; i+pr<size; i++)
                st[i][j] = op(st[i][j-1], st[i+pr][j-1]);
        }
    }
    int query(int l, int r){
        return op(st[l][LOGT[r-l]], st[r-(1 << LOG[r-l])+1][r]);    // +1 保证覆盖
    }
}
```

> - 以上以二目运算符为例，故相应建表、查询均为 $log_2$ 对数
> - *OI WIKI ST* 表：<https://oi-wiki.org/ds/sparse-table/>

# *Hashing*

##  *Hash Function*

-   *Hash* 散列/哈希：将任意类型值转换为关键码值
    -   *Hash Function* 哈希/散列函数：从任何数据中创建小的数字“指纹”的方法
        -   哈希函数应该尽可能使得哈希值均匀分布在目标空间中
            -   降维：将高维数据映射到低维空间
            -   数据应该低维空间中尽量均匀分布
        -   *Hash Value*：哈希值，哈希函数产生关键码值
        -   *Collision*：冲突，不同两个数据得到相同哈希值
    -   *Data Independent Hashing*：数据无关哈希（无监督）
        -   哈希函数基于某种概率理论
            -   对原始的特征空间作均匀划分
            -   对分布不均、有趋向性的数据集时，可能会导致高密度区域哈希桶臃肿，降低索引效率
    -   *Data Dependent Hashing*：数据依赖哈希（有监督）
        -   通过学习数据集的分布从而给出较好划分的哈希函数
            -   得到针对数据密度动态划分的哈希索引
            -   破坏了传统哈希函数的数据无关性，索引不具备普适性

-   *Hash* 应用
    -   提升查找效率：简单哈希函数主要用于构建哈希表
        -   要求哈希函数的降维、缩小查找空间性质
        -   计算简单、效率高
    -   信息安全方向：复杂哈希函数主要用于信息提取
        -   要求哈希函数的信息提取不可逆、非单调映射
            -   文件检验
            -   数字签名
            -   鉴权协议
        -   查表哈希
            -   *CRC* 系列算法：本身不是查表，但查表是其最快实现
            -   *Zobrist Hashing*
        -   混合哈希：利用以上各种方式
            -   *MD5*
            -   *Tiger*

### 哈希函数

-   哈希函数
    -   单值输入哈希函数：输入为确定范围、长度的内容，常用于提升查找效率，或者对前序哈希结果再次映射为更小范围的最终哈希值
        -   直接寻址法：取关键字、或其某个线性函数值 $hash(key) = (a * key + b) \% prime$
            -   $prime$：一般为质数，以使哈希值尽量均匀分布，常用的如：$2^{32}-5$
        -   数字分析法：寻找、利用数据规律构造冲突几率较小者
            -   如：生日信息前 2、3 位大体相同，冲突概率较大，优先舍去
        -   平方取中法：取关键字平方后中间几位
        -   折叠法：将关键字分割为位数相同部分，取其叠加和
        -   随机数法：以关键字作为随机数种子生成随机值
            -   适合关键字长度不同场合
    -   序列输入哈希函数：输入为不定长序列
        -   加法哈希
        -   位运算哈希
        -   乘法哈希

####    加法哈希

-   加法哈希：将输入元素相加得到哈希值
    -   标准加法哈希
        ```python
        AddingHash(input):
            hash = 0
            for ele in input:
                hash += ele
            # prime 为任意质数，常用 2^32 - 5
            hash = hash  % prime
        ```
        -   最终哈希结果 $\in [0, prime-1]$

####    位运算哈希

-   位运算哈希：利用位运算（移位、异或等）充分混合输入元素
    -   标准旋转哈希
        ```python
        RotationHash(input):
            hash = 0
            for ele in input:
                hash = (hash << 4) ^ (hash >> 28) ^ ele
            return hash % prime
        ```
    -   变形 1
        ```python
        hash = (hash<< 5) ^ (hash >> 27) ^ ele
        ```
    -   变形2
        ```python
        hash += ele
        hash ^= (hash << 10)
        hash ^= (hash >> 6)
        ```
    -   变形3
        ```python
        if (ele & 1) == 0:
            hash ^= (hash << 7) ^ ele ^ (hash >> 3)
        else:
            hash ^= ~((hash << 11) ^ ele ^ (hash >> 5))
        ```
    -   变形4
        ```python
        hash += (hash << 5) + ele
        ```
    -   变形5
        ```python
        hash = ele + (hash << 6) + (hash >> 16) - hash
        ```
    -   变形6
        ```python
        hash ^= (hash << 5) + ele + (hash >> 2)
        ```

####    乘法哈希

-   乘法哈希：利用乘法的不相关性
    -   平方取头尾随机数生成法：效果不好
    -   *Bernstein* 算法
        ```python
        Bernstein(input):
            hash = 0
            for ele in input:
                hash = 33 * hash + ele
            return hash
        ```
        > - 其他常用乘数：31、131、1313、13131、131313
    -   32位 *FNV* 算法
        ```python
        M_SHIFT =
        M_MASK =
        FNVHash(input):
            hash = 2166136261;
            for ele in input:
                hash = (hash * 16777619) ^ ele
            return (hash ^ (hash >> M_SHIFT)) & M_MASK
        ```
    -   改进的 *FNV* 算法
        ```python
        FNVHash_2(input):
            hash = 2166136261;
            for ele in input:
                hash = (hash ^ ele) * 16777619
            hash += hash << 13
            hash ^= hash >> 7
            hash += hash << 3
            hash ^= hash >> 17
            hash += hash << 5
            return hash
        ```
    -   乘数不固定
        ```python
        RSHash(input):
            hash = 0
            a, b = 378551, 63689
            for ele in input:
                hash = hash * a + ele
                a *= b
            return hash & 0x7FFFFFFF
        ```

> - 除法也类似乘法具有不相关性，但太慢


##  *Hashing Table*

-   哈希表/散列表：可根据哈希值直接访问的数据结构
    -   原理：以键计算得到的哈希值做为地址，缩小搜索空间、提高查找效率
        -   使用哈希函数为每个键计算哈希值，得到位于 $0, \cdots, m-1$ 之间整数
        -   按照哈希值把键分布在 $H[0, \cdots, m-1]$ 哈希表中
        -   查找匹配键时，以查找键哈希值作为 **起点** 在哈希表中搜索（存在冲突）
    -   冲突：不同关键字得到的哈希值相同
        -   不可避免，只能环节
    -   哈希表设计核心：降低冲突概率，尽量把键尽量均分在哈希表中
        -   散列表长度：常为质数（方便双散列）
        -   哈希函数
        -   冲突处理方案：根据冲突键存储方案可分为
            -   开散列/分离链：哈希表作为目录，使用额外数据空间组织哈希键
            -   闭散列/开放寻址：所有键存储在散列表本身中，不扩展存储空间
    -   应用
        -   字典/映射实现
        -   哈希连接

### *Load Factor*

$$\alpha = \frac {noempty} {m}$$

> - $m$：哈希表中 slots 数量（哈希表长度）
> - $noempty$：非空 slots 数量

-   负载因子
    -   闭散列：负载因子反映哈希表冲突可能性、查找效率
        -   负载因子过小：冲突可能性小，查找效率高，但浪费空间
        -   负载因子过大：冲突可能性大，查找效率低，空间利用率高
        -   负载因子取值最大为 1
        -   应适当平衡负载因子，负载因子接近1时重散列，避免冲突过多影响查找效率
    -   开散列：负载因子反映查找效率
        -   但应该无法反映冲突可能性（也无必要）
            -   开散列往往被用于应对大规模数据，冲突总是存在
            -   查找效率更多取决于数据（哈希值）偏倚程度
        -   负载因子可以大于 1

> - Java 中 `HashMap` 初始负载值为 0.75

### *Open Addressing*

$$
H_i = (hash(K) + d_i) mod m, i=1,2,\cdots
$$

-   闭散列、开放寻址：所有键存储在散列表本身中，不扩展存储空间
    -   哈希表 $m$ 至少要和哈希键数量 $n$ 取值范围一样大
    -   常用简单哈希函数
        -   平方取中法：平方后取中间部分（与数值各位均有关）
        -   除余法法：对哈希表长取模
        -   相乘取整法：乘以指定小数后取小数部分乘以哈希表长取整
        -   随机数法
    -   冲突解决：碰撞发生后，根据一定规则对原哈希值修正 $d_i$
        -   $d_i = i$：*linear probing*，线性探查
        -   $d_i = i^2, -i^2$：*quadratic probing*，二次探查
        -   $d_i = pseudo-random(key)$：伪随机探查
        -   $d_i = i * hash_2(K), i=0,1,2,\cdots$：*double hashing* 再哈希法
    -   说明
        -   冲突后探查机制可能无法探查整个哈希表，如：再散列法中 $hash_2(K)$ 与表长 `m` 不互质时
        -   散列表接近满时，连续单元格被占据（聚合），线性探查等方法性能恶化，在新插入元素后可能导致更大程度聚合
        -   再哈希法理论上性能较优，因其修正值基于原键值 $K$ 计算得出，密集冲突后修正值不同
-   算法流程
    -   插入：依次检查哈希值 $h(K)$、探查目标序列，直至找到空单元格放置键
    -   查找：给定查找键 $K$，计算哈希值 $h(K)$、探查目标序列，比较 $K$ 和单元格中键值
        -   若查找到匹配键，查找成功
        -   遇到空单元格，查找失败
    -   删除：**延迟删除**，用特殊符号标记曾经被占用过、现被删除的位置
        -   不能直接删除，否则的中间出现空单元格，影响查找正确性
-   算法效率
    -   成功查找访问次数：$S \approx \frac 1 2 (1+\frac 1 {(1-\alpha)})$
    -   失败查找访问次数：$U \approx \frac 1 2 [1+\frac 1 {(1-\alpha)^2}]$
    -   说明
        -   简化版本近似结论（散列规模越大，近似结论越正确）
        -   再哈希法数学分析困难，经验表明优秀的散列函数（两个），性能较线性探查好
        -   所有方法均无法避免散列表趋满时性能恶化

> - <https://www.cnblogs.com/anzhi/p/7536460.html>

### *Chaining*

-   开散列、拉链法、分桶法：哈希表作为目录项存储指向哈希桶的指针，哈希桶中存储哈希键
    -   目录项表：顺序表，连续存储空间，其中元素指向哈希桶
        -   可以通过哈希值在常数时间内定位：一般其索引位置就是哈希值
        -   目录项越多，数据分布相对越稀疏、碰撞概率越小、效率越高
    -   哈希桶：存储具有相同哈希值元素的线性表
        -   目录项指向顺序表：每个链即为哈希桶
        -   目录项指向顺序表链：链中每个顺序表为哈希桶
            -   即每个目录项对应多个哈希值，链接多个哈希桶
    -   说明
        -   理论上哈希桶可为链表，实务中内存总是按块申请，即需考虑桶分裂、串联
-   算法流程
    -   查找
        -   对查找键K，使用同样散列函数计算键散的函数值 $hash(K)$
        -   遍历相应单元格附着链表，查找是否存在键 $K$
    -   插入：计算键对应桶，在链表尾部添加键即可
    -   删除：查找需要删除的键，从链表中移除即可
-   算法效率
    -   效率取决于链表长度，而链表长度取决于字典、散列表长度和散列函数质量
        -   成功查找需要检查指针次数 $S = 1 + \alpha / 2$
        -   不成功查找需要检查指针次数 $U = \alpha$
        -   计算散列函数值是常数时间操作
        -   若键数 $n$、表长 $m$ 大致相等，平均情况下 $\in \Theta(1)$
    -   算法查找的高效是以额外空间为代价的

##  *Dynamic Hashing*

-   动态哈希：在哈希表中元素增加同时，动态调整哈希桶数目
    -   解决哈希表扩展问题
        -   哈希桶大小扩展：实务中内存总是按块申请，桶分裂后分块桶需串联
        -   哈希表长度扩展：桶深度增加查找效率下降，需调整哈希值精度（截断取模位数）
    -   开散列法在大规模、在线数据的扩展
        -   在原哈希表基础上进行动态桶扩展

### 多哈希表

-   多哈希表：任意哈希桶满时新建哈希表
    -   思想、实现简单
    -   占用空间大，数据分布偏斜程度较大时，桶利用率不高
    -   未解决桶深度大时查找效率下降问题
-   算法流程：操作时需要考虑多个哈希表
    -   插入
        -   若存在哈希相应桶中存在空闲区域，直接插入
            ![multi_hash_table_ori](imgs/multi_hash_table_ori.png)
        -   否则分裂，新建哈希表，插入元素至空闲区域
            ![multi_hash_table_splited](imgs/multi_hash_table_splited.png)
    -   查找：需要查找所有哈希表相应桶才能确定
        -   当表中元素较多时，可以考虑并行执行查找操作
    -   删除操作：若删除元素导致某哈希表空，可考虑删除该表

### 可扩展动态哈希

-   可扩展动态哈希：维护全局、局部桶深度，桶满时分裂全部目录项表、仅溢出桶
    -   分裂时代价较小
        -   翻倍目录项替代翻倍整个哈希表
        -   每次只分裂将要溢出桶
        -   只需要进行局部重散列，重分布需要分裂的桶
    -   目录项指数级增长
        -   多个目录项可能指向同一个桶
        -   数据分布不均时，会使得目录项很大

####    算法流程

> - `D`：全局位深度，hash值截断长度，为局部桶深度最大值
> - `L_i`：桶局部深度，等于指向其目录项数目

-   插入
    -   若对应桶存在空闲位，则直接插入
        ![dynamic_scalable_hash_table_ori](imgs/dynamic_scalable_hash_table_ori.png)
    -   否则分裂桶：分裂后两桶局部深度加 1
        ![dynamic_scalable_hash_table_splited](imgs/dynamic_scalable_hash_table_splited.png)
    -   若分裂桶局部深度不大于全局位深度
        -   创建新桶
        -   重散列原始桶中数据
        -   更新目录项中对应指针：分别指向分裂后桶
    -   若分类桶局部深度大于全局位深度
        -   更新全局位深度
        -   目录项翻倍
        -   创建新桶
        -   重散列原始桶中数据
        -   更新目录项中对应指针
            -   （新增）无关目录项仍然指向对应桶
            -   相关目录项指向分别指向分裂后桶
-   查找
    -   按照全局位深度截断哈希值在目标项表中查找目标项
    -   在对应桶中进行比较、查找
-   删除
    -   计算原始哈希值，按照全局位深度截断
    -   寻找相应目录项，找到对应桶，在桶中进行比较、删除
        -   若删除后发现桶为空，考虑与其兄弟桶合并，并使局部深度减 1

### 线性散列

-   线性散列：按次序分裂桶，**保证整个建表过程类似完全二叉树**
    -   整个哈希表建表过程 **始终保持为完全二叉树**
        -   每次分裂的桶是完全二叉树编号最小的叶子节点
        -   分裂前后桶间均为有序
    -   相较于可扩展动态哈希
        -   无需存放数据桶指针的专门目录项，节省空间
        -   能更自然的处理数据桶满的情况
        -   允许更灵活的选择桶分裂时机
        -   但若数据散列后分布不均，则问题可能比可扩散散列严重
    -   实现相较而言更复杂

####    算法流程

> - `N`：哈希表中初始桶数目，应为 2 的幂次
> - `d = log_2N`：表示桶数目需要位数
> - `level`：分裂轮数，初始值为 0，则每轮初始桶数为 $N * 2^{level}$
> - `Next`：下次要发生分裂的桶编号

![linear_hash_ori](imgs/linear_hash_ori.png)

-   桶分裂
    -   每次同分裂条件可以灵活选择
        -   设置桶填充因子，桶中记录数达到该值时进行分裂
        -   桶满时发生分裂
    -   每次发生的分裂的桶总是由 `Next` 决定
        ![linear_hash_splited_bucket](imgs/linear_hash_splited_bucket.png)
        -   与当前被插入的桶溢出无关，可引入溢出页处理桶溢出
        -   每次只分裂 `Next` 指向的桶，桶分裂后 `Next += 1`
        -   后续产生映像桶总是位于上次产生映像桶之后
    -   “轮转分裂进化”：各桶轮流进行分裂，一轮分裂完成后进入下轮分裂
        ![linear_hash_splited_level](imgs/linear_hash_splited_level.png)
-   查找
    -   根据 `N`、`level` 计算当前 `d` 值，截取原始哈希值
    -   若哈希值位于 `Next`、`N` 之间，说明该轮对应桶还未分裂，直接在桶中查找
    -   若哈希值小于 `Next`，说明该轮对应桶已经分裂，哈希值向前多取一位，在对应桶中查找
-   删除：插入操作的逆操作
    -   若删除元素后溢出块为空，可直接释放
    -   若删除元素后某个桶元素为空，`Next -= 1`
        -   当 `Next` 减少到 0，且最后桶也是空时 `Next = N/2 - 1`，同时 `level -= 1`

### *Perfect Hashing*

![hash_perfect_structure](imgs/hash_perfect_structure.png)

-   完美哈希：采用两级全域哈希，目录项链接独立哈希表的拉链哈希表
    -   二级哈希表开头部分存储哈希表元信息
        -   $m = n^2$：哈希表槽数，$n$ 为映射至该槽元素数量（此时由全域哈希性质：冲突次数期望小于 0.5）
        -   $a, b$：全域哈希参数
    -   算法复杂度
        -   时间复杂度：最坏情况下查找为 $O(1)$
        -   空间复杂度：期望空间为线性 $E(\sum_{i=1}^{m-1} \theta(n_i^2) = \theta(n)$

> - 完美哈希没有冲突的概率至少为 0.5

### 其他处理方案

-   其他难以清晰分类的方案
    -   *Multi Hashing* 多重哈希：使用一组哈希函数 $h_0,\cdots,h_n$ 依次计算哈希值，确定插入、查找地址
        -   类似增量类型方法，仅各次地址独立使用哈希函数计算
    -   *Rehashing* 重散列：扫描当前表，将所有键重新放置在更大的表中
        -   散列表趋满时唯一解决办法
    -   *Overflow Area* 建立公共溢出区：将哈希表分为基本表、溢出表两部分
        -   将发生冲突的元素都放入溢出区
    -   基本表中可以考虑为为每个哈希值设置多个slots
        -   即基本表直接存储哈希桶

![hash_overflow_area](imgs/hash_overflow_area.png)

##  *Universal Hashing*

-   全域哈希：键集合 $U$ 包含 $n$ 个键、哈希函数族 $H$ 中哈希函数 $h_i: U \rightarrow 0..m$，若 $H$ 满足以下则为全域哈希
    $$ \forall x \neq y \in U, | \{h|h \in H, h(x) = h(y) \} | = \frac {|H|} m $$
    > - $|H|$：哈希函数集合 $H$ 中函数数量
    -   独立与键值随机从中选择哈希函数，避免发生最差情况
    -   可利用全域哈希构建完美哈希

-   全域哈希性质
    -   全域哈希 $H$ 中任选哈希函数 $h_i$，对任意键 $x \neq y \in U$ 冲突概率小于 $\frac 1 m$
        -   由全域哈希函数定义，显然
    -   全域哈希 $H$ 中任选哈希函数 $h_i$，对任意键 $x \in U$，与其冲突键数目期望为 $\frac n m$，即 $E_{[collision_x]}=\frac n m$
        $$\begin{align*}
        E(C_x) &= E[\sum_{y \in U - \{x\}} C_{xy}] \\
            &= \sum_{y \in U - \{x\}} E[C_{xy}] \\
            &= \sum_{y \in U - \{x\}} \frac 1 m \\
            &= \frac {n-1} m
        \end{align*}$$
        > - $C_x$：任选哈希函数，与 $x$ 冲突的键数量
        > - $C_{xy} = \left \{ \begin{matrix} 1, & h_i(x) = h_i(y) \\ 0, & otherwise \end{matrix} \right.$：指示 $x,y$ 是否冲突的指示变量
        -   $m = n^2$ 时，冲突期望小于 0.5
            -   $n$ 个键两两组合数目为 $C_n^2$
            -   则 $E_{total} < C_n^2 \frac 1 n < 0.5$

-   以下构造 $[0,p-1] \rightarrow [0,m-1]$ 全域哈希
    -   $p$ 为足够大素数使得所有键值 $\in [0,p-1]$
        -   记 $Z_p = \{ 0,1,\cdots,p-1 \}$
        -   记 $Z_p^{*}=\{ 1,2,\cdots,p-1 \}$
        -   且哈希函数映射上限（哈希表长度） $m < max(U) < p$
    -   记哈希函数
        $$ \forall a \in Z_p^{*}, b \in Z_p, h_{a, b}(k) = ((a k + b) \% p) \% m $$
    -   则以下哈希函数族即为全域哈希
        $$ H_{p,m} = {h_{a,b}|a \in Z_p^{*}, b \in Z_p} $$

##  *Locality Sensitive Hashing*

-   *LSH* 局部敏感哈希：$(r_1,r_2,P_1,P_2)-sensitive$ 哈希函数族 $H$ 需满足如下条件
    $$\begin{align*}
    Pr_{H}[h(v) = h(q)] \geq P_1, & \forall q \in B(v, r_1) \\
    Pr_{H}[h(v) = h(q)] \geq P_2, & \forall q \notin B(v, r_2) \\
    \end{align*}$$
    > > -   $h \in H$
    > > -   $r_1 < r_2, P_1 > P_2$：函数族有效的条件
    > > -   $B(v, r)$：点 $v$ 的 $r$ 邻域
    > > -   $r_1, r_2$：距离，强调比例时会表示为 $r_1 = R, r_2 = cR$

### *LSH* 查找

![general_lsh_comparsion](imgs/general_lsh_comparsion.png)

-   *LSH* 查找：利用局部敏感哈希的性质，快速寻找和近似目标项
    -   *LSH* 中 相似目标（距离小）有更大概率发生冲突，相似目标更有可能映射到相同哈希桶中
        -   则只需要在目标所属的哈希桶中进行比较、查找即可
        -   无需和全集数据比较，大大缩小查找空间
        -   可视为降维查找方法
            -   将高维空间数据映射到 1 维空间，寻找可能近邻的数据点
            -   缩小范围后再进行精确比较
    -   概率放大：期望放大局部敏感哈希函数族 $Pr_1, Pr_2$ 之间差距
        -   增加哈希值长度（级联哈希函数中基本哈希函数数量） $k$
            -   每个哈希函数独立选择，则对每个级联哈希函数 $g_i$ 有 $Pr[g_i(v) = g_i(q)] \geq P_1^k$
            -   虽然增加哈希键位长会减小目标和近邻碰撞的概率，但同时也更大程度上减少了和非近邻碰撞的概率、减少搜索空间
            > - 级联哈希函数返回向量，需要对其再做哈希映射为标量，方便查找
        -   使用多个级联哈希函数分别处理待搜索目标，增加级联哈希函数数量（哈希表数量） $L$
            -   $L$ 个哈希表中候选项包含真实近邻概率 **至少** 为 $1 - (1 - P_1^k)^L$
            -   在 $L$ 个哈希表分别寻找落入相同哈希桶个体作为候选项
                -   增加哈希表数量能有效增加候选集包含近邻可能性
                -   但同时也会增大搜索空间

### 基于汉明距离的 *LSH*

-   哈希函数族 $H = \{ h_1, h_2, \cdots, h_n \}$
    -   其中函数 $h_i$ 为 $\{0, 1\}^M$ 到 $\{0, 1\}$ 的映射，作用为随机返回特定比特位上的值
    -   从 $H$ 中随机的选择哈希函数 $h_i$
        -   则 $Pr[h_i(v) = h_i(q)]$ 等于 $v, q$ 相同比特数比例，则
            -   $Pr_1 = 1 - \frac R d$
            -   $Pr_2 = 1 - \frac {cR} M$
        -   考虑到 $Pr_1 > Pr_2$，即此哈希函数族是局部敏感的
-   说明
    -   适用于定长 $M$ 二进制序列数据的近似搜索
    -   在汉明距离空间中搜索近邻，要求数据为二进制表示
    -   其他距离需要嵌入汉明距离空间才能使用
        -   欧几里得距离没有直接嵌入汉明空间的方法
            -   一般假设欧几里得距离和曼哈顿距离差别不大
            -   直接使用对曼哈顿距离保距嵌入方式

### 基于 *Jaccard* 系数的 *LSH*

-   *Min-hashing* 函数族：对矩阵 $A$ 进行行随机重排 $\pi$，定义 *Min-hashing* 如下
    $$h_{\pi}(C) = \min \pi(C)$$
    > - $C$：列，表示带比较集合
    > - $\min \pi(C)$：$\pi$ 重排矩阵中 $C$ 列中首个 1 所在行数
    -   则不同列（集合） *Min-hashing* 相等概率等于二者 *Jaccard* 系数
        $$\begin{align*}
        Pr(h_{\pi}(C_1)  = h_{\pi}(C_2)) & = \frac a {a + b} \\
        & = Jaccard_d(C_1, C_2)
        \end{align*}$$
        > - $a$：列 $C_1, C_2$ 取值均为 1 的行数
        > - $b$：列 $C_1, C_2$ 中仅有一者取值为 1 的行数
        > - 根据 *Min-hashing* 定义，不同列均取 0 行被忽略

-   说明
    -   适用于定长 $M$ 二进制序列数据的近似搜索
        -   将待比较数据拼接为 0、1 构成矩阵 $A \in R^{M * N}$
        -   则寻找相似集合，即寻找矩阵中相似列
    -   用 *Jaccard* 系数代表集合（序列数据）间相似距离，用于搜索近邻

####    实现说明

-   *Min-hashing* 实现说明
    -   数据量过大时，对行随机重排仍然非常耗时，考虑使用哈希函数模拟行随机重排
        -   每个哈希函数对应一次随机重排
            -   哈希函数视为线性变换
            -   然后用哈希函数结果对总行数取模
        -   原行号经过哈希函数映射即为新行号
    -   为减少遍历数据次数，考虑使用迭代方法求解
        ```c
        for i from 0 to N-1:
            for j from 0 to M-1:
                if D[i][j] == 1:
                    for k from 1 to K:
                        # 更新随机重拍后，第 `j` 列首个 1 位置
                        DD[k][j] = min(h_k(i), DD[k][j])
        ```
        > - $D$：原始数据特征矩阵
        > - $DD$：$Min-hashing* 签名矩阵
        > - $N$：特征数量，原始特征矩阵行数
        > - $M$：集合数量，原始特征矩阵列数
        > - $K$：模拟的随机重排次数，*Min-hashing* 签名矩阵行数
        > - $h_k,k=1,...,K$：$K$ 个模拟随机重排的哈希函数，如 $h(x) = (2x + 7) mod N$
        -   初始化 *Min-hashing* 签名矩阵所有值为 $\infty$
        -   遍历 $N$ 个特征、$M$ 个集合
            -   查看每个对应元素是否为 1
            -   若元素为 1，则分别使用 $K$ 个哈希函数计算模拟重排后对应的行数
            -   若计算出行数小于当前 *Min-hash* 签名矩阵相应哈希函数、集合对应行数，更新
        -   遍历一遍原始数据之后即得到所有模拟重排的签名矩阵

### *Exact Euclidean LSH*

-   $E^2LSH$ 欧式局部LSH：利用 *p-stable* 分布性质将欧式距离同哈希值相联系，实现局部敏感
    -   $E^2LSH$ 特点
        -   基于概率模型生成索引编码结果不稳定
        -   随编码位数 $k$ 增加的，准确率提升缓慢
        -   级联哈希函数数量 $L$ 较多时，需要大量存储空间，不适合大规模数据索引
    -   对动态变化的数据集，固定哈希编码的局部敏感哈希方法对数据 **动态支持性有限**，无法很好的适应数据集动态变化
        -   受限于初始数据集分布特性，无法持续保证有效性
        -   虽然在原理上支持数据集动态变化，但若数据集大小发生较大变化，则其相应哈希参数（如哈希编码长度）等需要随之调整，需要从新索引整个数据库

####    *P-stable* 哈希函数族

-   *p-stable* 哈希函数族
    $$ h_{a, b}(v) = \lfloor \frac {av + b} r \rfloor $$
    > - $v$：$n$ 维特征向量
    > - $a = (X_1,X_2,\cdots,X_n)$：其中分量为独立同 *p-stable* 分布的随机变量
    > - $b \in [0, r]$：均匀分布随机变量

-   *p-stable* 哈希函数碰撞概率：考虑$\|v_1 - v_2\|_p = c$的两个样本碰撞概率
    -   显然，仅在 $|av_1 - av_2| \leq r$ 时，才存在合适的 $b$ 使得 $h_{a,b}(v_1) = h_{a,b}(v_2)$
        -   即两个样本碰撞，不失一般性可设 $av_1 \leq av_2$
        -   此 $r$ 即代表局部敏感的 **局部范围**
    -   若 $(k-1)r \leq av_1 \leq av_2 < kr$，即两个样本与 $a$ 内积在同一分段内
        -   易得满足条件的 $b \in [0,kr-av_2) \cup [kr-av_1, r]$
        -   即随机变量 $b$ 取值合适的概率为 $1 - \frac {av_2 - av_1} r$
    -   若 $(k-1)r \leq av_1 \leq kr \leq av_2$，即两个样本 $a$ 在相邻分段内
        -   易得满足条件的 $b \in [kr-av_1, (k+1)r-av_2)$
        -   即随机变量 $b$ 取值合适的概率同样为 $1 - \frac {av_2 - av_1} r$
    -   考虑 $av_2 - av_1$ 分布为 $cX$，则两样本碰撞概率为
        $$\begin{align*}
        p(c)  & = Pr_{a,b}(h_{a,b}(v_1) = h_{a,b}(v_2)) \\
        & = \int_0^r \frac 1 c f_p(\frac t c)(1 - \frac t r)dt
        \end{align*}$$
        > - $c = \|v_1 - v_2\|_p$：特征向量之间$L_p$范数距离
        > - $t = a(v_1 - v_2)$
        > - $f$：p稳定分布的概率密度函数
        -   $p=1$ 柯西分布
            $$ p(c) = 2 \frac {tan^{-1}(r/c)} \pi - \frac 1 {\pi(r/c)} ln(1 + (r/c)^2) $$
        -   $p=2$ 正态分布
            $$ p(c) = 1 - 2norm(-r/c) - \frac 2 {\sqrt{2\pi} r/c} (1 - e^{-(r^2/2c^2)}) $$

####    实现说明

-   近邻碰撞概率限制
    -   $r$ 最优值取决于数据集、查询点
        -   根据文献，建议 $r = 4$
    -   若要求近邻 $v \in B(q,R)$ 以不小于 $1-\sigma$ 概率碰撞，则有
        $$\begin{align*}
        1 - (1 - p(R)^k)^L & \geq 1 - \sigma \\
        \Rightarrow L & \geq \frac {log \sigma} {log(1 - p(R)^k)}
        \end{align*}$$
        则可取
        $$ L = \lceil \frac {log \sigma} {log(1-p(R)^k)} \rceil $$
    -   $k$ 最优值是使得 $T_g + T_c$ 最小者
        -   $T_g = O(dkL)$：建表时间复杂度
        -   $T_c = O(d |collisions|)$：精确搜索时间复杂度
        -   $T_g$、$T_c$ 随着 $k$ 增大而增大、减小

> - 具体实现参考<https://www.mit.edu/~andoni/LSH/manual.pdf>

-   搜索空间限制：哈希表数量 $L$ 较多时，所有碰撞样本数量可能非常大，考虑只选择 $3L$ 个样本点
    -   此时每个哈希键位长 $k$、哈希表数量 $L$ 需保证以下条件，则算法正确
        -   若存在 $v^{ * }$ 距离待检索点 $q$ 距离小于 $r_1$，则存在 $g_j(v^{ * }) = g_j(q)$
        -   与 $q$ 距离大于 $r_2$、可能和 $q$ 碰撞的点的数量小于 $3L$
            $$ \sum_{j=1}^L |(P-B(q,r_2)) \cap g_j^{-1}(g_j(q))| < 3L $$
    -   可以证明，$k, L$ 取以下值时，以上两个条件以常数概率成立（此性质是局部敏感函数性质，不要求是 $E^2LSH$）
        $$\begin{align*}
        k & = log_{1/p_2} n\\
        L & = n^{\rho} \\
        \rho & = \frac {ln 1/p_1} {ln 1/p_2}
        \end{align*}$$
    -   $\rho$ 对算法效率起决定性作用
        -   且有定理：距离尺度 $D$ 下，若 $H$ 为 $(R,cR,p_1,p_2)$-敏感哈希函数族，则存在适合 *(R,c)-NN* 的算法，其空间复杂度为 $O(dn + n^{1+\rho})$、查询时间为 $O(n^{\rho})$ 倍距离计算、哈希函数计算为 $O(n^{\rho} log_{1/p_2}n)$， 其中 $\rho = \frac {ln 1/p_1} {ln 1/p_2}$
        -   $r$ 足够大、充分远离 0 时，$\rho$ 对其不是很敏感
        -   $p_1, p_2$ 随 $r$ 增大而增大，而 $k = log_{1/p_2} n$ 也随之增大，所以 $r$ 不能取过大值

##  *GeoHash*

![geohash_demo](imgs/geohash_demo.png)

-   *GeoHash*：将经纬度按二分法、递归的编码为 *Base32* 字符串（二进制串）
    -   编码逻辑
        -   将经度、纬度各自用二分法二进制编码
            -   将经度 $[-180,180]$ 二分为 $[-180,0),[0,180]$，左 0、右 1（维度类似）
            -   根据目标点经度确定所属区间范围，得到首位编码 0、1
            -   将所在区间范围继续划分，重复、得到后续编码，直至得到期望长度编码
        -   从 0 位开始，偶数位、奇数位分别放置经度、维度，混编为二进制位串
            -   经纬度混编可确保临近点编码结果相近
        -   将二进制位串 5 位一组转换为 *Base32* 编码字符串
            -   *Base 32* 即按顺序的数字、字母（排除 `ailo`）
    -   *GeoHash* 编码结果代表一个区域，编码精度越高（串长）区域范围越小
        -   区域内所有点编码结果相同，可隐藏具体的位置信息
    -   可将 *Proximity Search* 临近搜索转换为字符串前缀匹配，提高计算效率
        -   但注意查找最近点时，需要同时搜索周围区块（*Base32* 划分的 32 个区块周围）
        -   *GeoHash* 将地球视为二维平面对经纬度二分，实际地球面实际距离、经纬度距离有差异（低纬度单位维度差异的距离差异较大）

> - *GeoHash*，一种高效的地理编码方式：<https://segmentfault.com/a/1190000042971576>

