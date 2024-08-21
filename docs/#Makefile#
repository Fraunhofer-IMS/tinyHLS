.DEFAULT_GOAL := help

all: pdf html ug-pdf ug-html
	mkdir -p public/img/
	cp -vr figures/* public/img/

# Generate PDF userguide
# removed for now. 	  -r asciidoctor-diagram
pdf:
	[ -f revnumber.txt ] && REVNUMBER='-a revnumber='"$$(cat revnumber.txt)" || unset REVNUMBER; \
	asciidoctor-pdf $$REVNUMBER \
	  -a allow-uri-read \
	  -a pdf-theme=neorv32-theme.yml \
	  ug/main.adoc \
	  --out-file public/pdf/tinyHLS.pdf

# Generate HTML userguide
# Removed for now: 	  -r asciidoctor-diagram
html:
	[ -f revnumber.txt ] && REVNUMBER='-a revnumber='"$$(cat revnumber.txt)" || unset REVNUMBER; \
	asciidoctor $$REVNUMBER \
	  ug/index.adoc \
	  --out-file public/index.html

# Generate DOXYGEN software documentation
# This will be part of a later version 
#doxygen:
#	doxygen Doxyfile

# Generate revnumber.txt for overriding the revnumber attribute in 'pdf' and/or 'html'
revnumber:
	if [ `git tag -l | grep nightly` ]; then git tag -d nightly; fi
	git describe --long --tags | sed 's#\([^-]*-g\)#r\1#;' > revnumber.txt
	cat revnumber.txt

# Build 'pdf' and 'html' in an 'asciidoctor-wavedrom' container
container: revnumber
	docker run --rm -v /$(shell pwd)://documents/ btdi/asciidoctor make all

# Help
help:
	@echo "Targets:"
	@echo " help    - show this text"
	@echo " pdf     - build userguide as pdf file (public/pdf/tinyHLS.pdf)"
	@echo " html    - build userguide as HTML page (public/index.html)"
	@echo " doxygen - build software documentation as HTML page (doxygen_build/html/index.html)"
