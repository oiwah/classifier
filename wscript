#!/usr/bin/python

APPNAME = 'classifier'
VERSION = '0.0.1'

top = '.'
out = 'build'

def options(opt):
    opt.tool_options('compiler_cxx')

def configure(conf):
    conf.check_tool('compiler_cxx')
    conf.env.append_value('CXXFLAGS', ['-O2', '-W', '-Wall'])
    conf.env.HPREFIX=conf.env.PREFIX+'/include/classifier'

def build(bld):
    bld.recurse('src')
