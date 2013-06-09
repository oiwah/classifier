#!/usr/bin/python

APPNAME = 'classifier'
VERSION = '0.5.0'

top = '.'
out = 'build'

def options(opt):
    opt.load('compiler_cxx')

def configure(conf):
    conf.load('compiler_cxx')
    conf.env.append_unique('CXXFLAGS',
                           ['-std=c++0x',
                            '-O2',
                            '-W',
                            '-Wall'])
    conf.env.append_unique('LINKFLAGS', ['-std=c++0x',
                                         '-O2',
                                         '-W',
                                         '-Wall'])
    conf.env.HPREFIX = conf.env.PREFIX + '/include/classifier'

def build(bld):
#    bld.recurse('binary')
    bld.recurse('multiclass')
    bld.recurse('test')
