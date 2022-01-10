from os import link
import numpy as np

class Node:
    def __init__(self, item):
        self.item = item
        self.next = None


class SingleLinkList:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None
    
    def length(self):
        cur = self.head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count
    
    def items(self):
        # Traverse
        cur = self.head
        while cur is not None:
            yield cur.item
            cur = cur.next
    
    def add(self, item):
        # add item to head
        node = Node(item)
        node.next = self.head
        self.head = node

    def append(self, item):
        # add item to tail
        node = Node(item)
        if self.is_empty():
            self.head = node
        else:
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            cur.next = node
        
    def insert(self, index, item):
        if index <=0:
            self.add(item)
        elif index > (self.length() - 1):
            self.append(item)
        else:
            node = Node(item)
            cur = self.head
            for i in range(index - 1):
                cur = cur.next
            node.next = cur.next
            cur.next = node
    
    def remove(self, item):
        cur = self.head
        pre = None
        while cur is not None:
            if cur.item == item:
                if not pre:
                    self.head = cur.next
                else:
                    pre.next = cur.next
                return True
            else:
                pre = cur
                cur = cur.next
    

if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 4, 4, 5, 6, 7, 3, 3], dtype=np.uint8)
    y = np.array([1, 2, 3, 3], dtype=np.uint8)
    z = np.array([3, 3, 4, 5])
    node1 = Node(x)
    node2 = Node(y)
    node3 = Node(z)
    link_list = SingleLinkList()
    link_list.head = node1
    node1.next = node2
    node2.next = node3
    # print(link_list.head.item)
    # print(link_list.head.next.item)
    # print(link_list.head.next.next.item)
    for n in link_list.items():
        print(n)