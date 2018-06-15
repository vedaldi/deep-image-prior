url = vgg@login.robots.ox.ac.uk:~/WWW/research/deepquiz

.PHONY: all, post

all:
	# echo "Nothing to do."

post:
	rsync -rav quiz.{html,css,js} $(url)/
	rsync -ravl images/ $(url)/images/