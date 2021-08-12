.PHONY: up
up:
	@docker run --rm -v `pwd`:/home/jovyan/work -p 8888:8888 --name jupyter jupyter/scipy-notebook

.PHONY: down
down:
	@docker rm -f jupyter