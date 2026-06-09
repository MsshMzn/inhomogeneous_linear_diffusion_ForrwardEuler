clean:
	rm -f diff-*

pull:
	git pull origin main

push: 
	git commit -a
	git push -u origin main
