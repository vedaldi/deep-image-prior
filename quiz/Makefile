url = vgg@login.robots.ox.ac.uk:~/WWW/research/deepquiz

.PHONY: all, post

all:
	# echo "Nothing to do."

poster.pdf: poster.tex
	pdflatex poster.tex

montage.png:
	montage -tile 10x -label %t images/*.png  montage.png

post:
	rsync -rav index.html quiz.{css,js} $(url)/
	rsync -ravl images/ $(url)/images/
