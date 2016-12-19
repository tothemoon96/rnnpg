all:
	$(MAKE) -C rnnpg
	$(MAKE) -C rnnpg-decoder
	$(MAKE) -C first-sentence-generator
	$(MAKE) -C rnnpg-generator

clean:
	$(MAKE) -C rnnpg clean
	$(MAKE) -C rnnpg-decoder clean
	$(MAKE) -C first-sentence-generator clean
	$(MAKE) -C rnnpg-generator clean
