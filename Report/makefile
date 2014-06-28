FILE=report

all: $(FILE).pdf

.PHONY: clean
clean:
	\rm *.aux *.blg *.bbl *.log *.out

$(FILE).pdf: $(FILE).tex
	pdflatex $(FILE)
	pdflatex $(FILE)
	bibtex $(FILE)
	pdflatex $(FILE)
	pdflatex $(FILE)
