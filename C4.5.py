# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:06:13 2018

@author: Administrator
"""

class Node:  
    '''''Represents a decision tree node. 
     
    '''  
    def __init__(self, parent = None, dataset = None):  
        self.dataset = dataset # 落在该结点的训练实例集  
        self.result = None # 结果类标签  
        self.attr = None # 该结点的分裂属性ID  
        self.childs = {} # 该结点的子树列表，key-value pair: (属性attr的值, 对应的子树)  
        self.parent = parent # 该结点的父亲结点  
          
  
  
def entropy(props):  
    if (not isinstance(props, (tuple, list))):  
        return None  
      
    from math import log  
    log2 = lambda x:log(x)/log(2) # 计算经验熵
    e = 0.0  
    for p in props:  
        e -= p * log2(p)  
    return e  
  
  
def info_gain(D, A, T = -1, return_ratio = False):  
    '''''特征A对训练数据集D的信息增益 g(D,A) 
     
    g(D,A)=entropy(D) - entropy(D|A) 
            假设数据集D的每个元组的最后一个特征为类标签 
    T为目标属性的ID，-1表示元组的最后一个元素为目标'''  
    if (not isinstance(D, (set, list))):  
        return None  
    if (not type(A) is int):  
        return None  
    C = {} # 结果计数字典  
    DA = {} # 属性A的取值计数字典  
    CDA = {} # 结果和属性A的不同组合的取值计数字典  
    for t in D:  
        C[t[T]] = C.get(t[T], 0) + 1  #统计目标属性各种取值下的个数，用户经验熵的计算
        DA[t[A]] = DA.get(t[A], 0) + 1  #统计属性列下各种取值的个数，用于计算经验条件熵
        CDA[(t[T], t[A])] = CDA.get((t[T], t[A]), 0) + 1  #统计（属性列，目标列）下各种组合取值的个数，例如（女，合格）（男、合格）（）
      
    PC = map(lambda x : x / len(D), C.values()) # 类别的概率列表
    entropy_D = entropy(tuple(PC)) # map返回的对象类型为map，需要强制类型转换为元组  
  
  
    PCDA = {} # 特征A的每个取值给定的条件下各个类别的概率（条件概率）  
    for key, value in CDA.items():  
        a = key[1] # 特征A的取值
        pca = value / DA[a]  
        PCDA.setdefault(a, []).append(pca)  
      
    condition_entropy = 0.0  
    for a, v in DA.items():  
        p = v / len(D)  
        e = entropy(PCDA[a])  
        condition_entropy += e * p  #计算经验条件熵
      
    if (return_ratio):  
        return (entropy_D - condition_entropy) / entropy_D  #C4.5的信息增益比
    else:  
        return entropy_D - condition_entropy  #ID3的信息增益
      
def get_result(D, T = -1):  
    '''''获取数据集D中实例数最大的目标特征T的值'''  
    if (not isinstance(D, (set, list))):  
        return None  
    if (not type(T) is int):  
        return None  
    count = {}  
    for t in D:  
        count[t[T]] = count.get(t[T], 0) + 1  
    max_count = 0  
    for key, value in count.items():  
        if (value > max_count):  
            max_count = value  
            result = key  
    return result   
  
  
def devide_set(D, A):  
    '''''根据特征A的值把数据集D分裂为多个子集'''  
    #判断D的数据类型是set和list类型
    if (not isinstance(D, (set, list))):  
        return None
    #判断A的数据类型是否是int型
    if (not type(A) is int):  
        return None  
    subset = {}
    '''''根据特征A的结果划分数据集'''
    for t in D:
        subset.setdefault(t[A], []).append(t)  
    return subset
  
  
def build_tree(D, A, threshold = 0.0001, T = -1, Tree = None, algo = "C4.5"):  
    '''''根据数据集D和特征集A构建决策树. 
     
    T为目标属性在元组中的索引 . 目前支持ID3和C4.5两种算法''' 
    #判断Tree是否存在和Tree是否是节点
    if (Tree != None and not isinstance(Tree, Node)):  
        return None
    #判断数据集D的类型是否是set集合和list集合的一种，如果不是直接返回
    if (not isinstance(D, (set, list))):  
        return None
    #判断特征集A的类型是否是一个set集合
    if (not type(A) is set):  
        return None  
      
    if (None == Tree):  
        Tree = Node(None, D)  
    subset = devide_set(D, T)   #根据特征T的取值拆分数据集  
    if (len(subset) <= 1):  #如果该特征T的取值为一个时，则这个唯一取值为这个节点的结果
        for key in subset.keys():  
            Tree.result = key  
        del(subset)  
        return Tree  
    if (len(A) <= 0):  #当特征个数小于等于0的时候，返回
        Tree.result = get_result(D)  
        return Tree  
    use_gain_ratio = False if algo == "ID3" else True  #是要实现ID3还是C4.5算法，如果是ID3算法，use_gain_ratio为false,否则为true
    max_gain = 0.0  
    for a in A:  
        gain = info_gain(D, a, return_ratio = use_gain_ratio)  
        if (gain > max_gain):  
            max_gain = gain  
            attr_id = a # 获取信息增益最大的特征  
    if (max_gain < threshold):  #判断信息增益比是否小于阈值，如果小于，返回数据集D中实例数最大的目标特征T的值
        Tree.result = get_result(D)  
        return Tree  
    Tree.attr = attr_id
    
    subD = devide_set(D, attr_id)  
    del(D[:]) # 删除中间数据,释放内存  
    Tree.dataset = None  
    A.discard(attr_id) # 从特征集中排查已经使用过的特征  
    for key in subD.keys():  
        tree = Node(Tree, subD.get(key))  
        Tree.childs[key] = tree  
        build_tree(subD.get(key), A, threshold, T, tree)  
    return Tree  
  
  
def print_brance(brance, target):  #输出结果
    odd = 0   
    for e in brance:          
        print(e, end = ('=' if odd == 0 else '∧'))  
        odd = 1 - odd  
    print("target =", target)  
  
  
def print_tree(Tree, stack = []):   
    if (None == Tree):  
        return  
    if (None != Tree.result):  
        print_brance(stack, Tree.result)  
        return
    stack.append(Tree.attr)  
    for key, value in Tree.childs.items():  
        stack.append(key)
        print_tree(value, stack)  
        stack.pop()  
    stack.pop()  

# =============================================================================
# #根据决策树产生数据结果   
# def classify(Tree, instance):  
#     if (None == Tree):  
#         return None  
#     if (None != Tree.result):  
#         return Tree.result  
#     return classify(Tree.childs[instance[Tree.attr]], instance)
# =============================================================================

#导入操作excel文件的xlrd库     
import xlrd
#读取文件
bk = xlrd.open_workbook("C:\\Users\\lenovo\\Desktop\\decisiontree_data.xls")
try:
    sh = bk.sheet_by_name("Sheet1")#假设数据在该文件的sheet1下，读取sheet1
except:
    print("当前文件不存在Sheet1")#如果不存在sheet1，则输出当前文件不存在sheet1
rowNum = sh.nrows #读取改文件sheet1下数据的行数
dataset = [] #定义存储数据的列表dataset
#按行循环读取数据
for i in range(1,rowNum):
    rowData = sh.row_values(i)
    dataset.append(rowData)
#开始建立决策树
T = build_tree(dataset, set(range(0, len(dataset[0]) - 1)))
#打印输出决策树
print_tree(T)
# =============================================================================
# #根据决策树产生结果
# print(classify(T, ('女', '好', '每周大于三小时', 'A', 'C', 'B', 'D', 'D')))
# =============================================================================
