PYTHON=python

site=doc/site/docs

mfiles=\
vl_argparse.m \
vl_nnconv.m \
vl_nndropout.m \
vl_nnloss.m \
vl_nnnoffset.m \
vl_nnnormalize.m \
vl_nnpool.m \
vl_nnrelu.m \
vl_nnsoftmax.m \
vl_nnsoftmaxloss.m \
vl_rootnn.m \
vl_setupnn.m \
vl_simplenn.m \
vl_simplenn_diagnose.m \
vl_simplenn_display.m \
vl_simplenn_move.m

mfile_src=$(addprefix matlab/src/, $(mfiles))
mfile_doc=$(patsubst %.m,%.md,$(addprefix $(site)/mfiles/, $(mfiles)))

$(warning $(mfile_doc))

all: $(mfile_doc)

# Create
$(site)/mfiles/.stamp:
	mkdir -p $(site)/mfiles/ ; touch $(site)/mfiles/.stamp

$(site)/mfiles/%.md : matlab/%.m $(site)/mfiles/.stamp doc/matdoc.py doc/matdocparser.py
	$(PYTHON) doc/matdoc.py "$(<)" > "$(@)"

