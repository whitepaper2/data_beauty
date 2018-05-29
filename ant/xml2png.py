#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 下午2:02
# @Author  : pengyuan.li
# @Site    : 
# @File    : xml2png.py
# @Software: PyCharm

from xml.parsers import expat
class Element(object):
    '''analyze a element'''
    def __init__(self, name, attributes):
        #record tag and attribute dictionary
        self.name = name
        self.attributes = attributes
        #clear the element cdata and its children
        self.cdata = ''
        self.children = [ ]

    def addChild(self, element):
        self.children.append(element)

    def getAttribute(self, key):
        return self.attributes.get(key)

    def getData(self):
        return self.cdata

    def getElements(self, name = ''):
        if name:
            return [ c for c in self.children if c.name == name ]
        else:
            return list(self.children)

class Xml2Obj(object):
    '''transform XML to Object'''
    def __init__(self):
        self.root = None
        self.nodeStack = [ ]
    def StartElement(self, name, attributes):
        'Expat start element event handler'
        #make instance of class
        element = Element(name.encode(), attributes)
        #put the element into stack and make it become child_element
        if self.nodeStack:
            parent = self.nodeStack[-1]
            parent.addChild(element)
        else:
            self.root = element
        self.nodeStack.append(element)

    def EndElement(self, name):
        'Expat end element event handler'
        self.nodeStack.pop()

    def CharacterData(self, data):
        '''Expat character data event handler'''
        if data.strip():
            data = data.encode()
            element = self.nodeStack[-1]
            element.cdata += data

    def Parse(self, filename):
        #create Expat analyzer
        Parser = expat.ParserCreate()
        #Set the Expat event handlers to our methods
        Parser.StartElementHandler = self.StartElement
        Parser.EndElementHandler = self.EndElement
        Parser.CharacterDataHandler = self.CharacterData
        #analyz XML file
        ParserStatus = Parser.Parse(open(filename).read(), 1)
        return self.root

parser = Xml2Obj()
root_element = parser.Parse('./test.xml')
print(root_element)