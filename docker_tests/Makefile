TAG=cherenkovdeconvolution_test

image: .IMAGE
.IMAGE: Dockerfile
	- docker rmi -f $(TAG)
	docker build --tag $(TAG) .
	echo "$(TAG)" > $@

clean:
	- docker rmi -f $(TAG)
	rm -f .IMAGE

.PHONY: image clean
