# file: matdoc.py
# author: Andrea Vedaldi
# brief: Extact comments from a MATLAB mfile and generate a Markdown file

import sys, os, re, shutil
import subprocess, signal

from matdocparser import *
from optparse import OptionParser

usage = """usage: %prog [options] <mfile>

Extracts the comments from the specified <mfile> and prints a Markdown
version of them."""

optparser = OptionParser(usage=usage)
optparser.add_option(
    "-v", "--verbose",
    dest    = "verb",
    default = False,
    action  = "store_true",
    help    = "print debug information")

# --------------------------------------------------------------------
def extract(path):
# --------------------------------------------------------------------
    """
    (BODY, FUNC, BRIEF) = extract(PATH) extracts the comment BODY, the
    function name FUNC and the brief description BRIEF from the MATLAB
    M-file located at PATH.
    """
    body         = []
    func         = ""
    brief        = ""
    seenfunction = False
    seenpercent  = False

    for l in open(path):
        line = l.strip().lstrip()
        if line.startswith('%'): seenpercent = True
        if line.startswith('function'):
            seenfunction = True
            continue
        if not line.startswith('%'):
            if (seenfunction and seenpercent) or not seenfunction:
                break
            else:
                continue
        # remove leading `%' character
        line = line[1:] #
        body.append('%s\n' % line)
    # Extract header from body
    if len(body) > 0:
        head  = body[0]
        body  = body[1:]
        match = re.match(r"^\s*(\w+)\s*(\S.*)\n$", head)
        func  = match.group(1)
        brief = match.group(2)
    return (body, func, brief)


class Frame(object):
    prefix = ""
    before = None
    def __init__(self, prefix, before = None):
        self.prefix = prefix
        self.before = before

class Context(object):
    frames = []
    def __str__(self):
        text =  ""
        for f in self.frames:
            if not f.before:
                text = text + f.prefix
            else:
                text = text + f.prefix[:-len(f.before)] + f.before
                f.before = None
        return text

    def pop(self):
        f = self.frames[-1]
        del self.frames[-1]
        return f

    def push(self, frame):
        self.frames.append(frame)

def render_L(tree, context):
    print "%s%s" % (context,tree.text)

def render_DH(tree, context):
    print "%s**%s** [*%s*]" % (context, tree.description.strip(), tree.inner_text.strip())

def render_DI(tree, context):
    context.push(Frame("    ", "*   "))
    render_DH(tree.children[0], context)
    print context
    if len(tree.children) > 1:
        render_DIVL(tree.children[1], context)
    context.pop()

def render_DL(tree, context):
    for n in tree.children: render_DI(n, context)

def render_P(tree, context):
    for n in tree.children: render_L(n, context)
    print context

def render_B(tree, context):
    print context

def render_V(tree, context):
    context.push(Frame("    "))
    for n in tree.children:
        if n.isa(L): render_L(n, context)
        elif n.isa(B): render_B(n, context)
    context.pop()

def render_BL(tree, context):
    for n in tree.children:
        context.push(Frame("    ", "+   "))
        render_DIVL(n, context)
        context.pop()

def render_DIVL(tree, context):
    for n in tree.children:
        if n.isa(P): render_P(n, context)
        elif n.isa(BL): render_BL(n, context)
        elif n.isa(DL): render_DL(n, context)
        elif n.isa(V): render_V(n, context)
        context.before = ""

def render(func, brief, tree):
    print "## `%s` - %s" % (func.upper(), brief)
    render_DIVL(tree, Context())

if __name__ == '__main__':
    (opts, args) = optparser.parse_args()
    if len(args) != 1:
        optparser.print_help()
        sys.exit(2)
    mfilePath = args[0]
    (body, func, brief) = extract(mfilePath)
    parser = Parser()
    lexer = Lexer(body)
    tree = parser.parse(lexer)
    render(func, brief, tree)
