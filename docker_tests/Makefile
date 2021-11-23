TAG=cherenkovdeconvolution_test

tests: .IMAGE
	docker run --rm $(TAG)
.IMAGE: .DEPS tests.py Dockerfile
	docker build --tag $(TAG) . && touch $@
.DEPS: ../CherenkovDeconvolution_jl/__init__.py ../LICENSE.md ../README.md ../setup.py
	mkdir -p $@
	cp -r ../CherenkovDeconvolution_jl ../LICENSE.md ../README.md ../setup.py $@ && touch $@

clean:
	- docker rmi -f $(TAG)
	rm -rf .IMAGE .DEPS

.PHONY: tests clean
