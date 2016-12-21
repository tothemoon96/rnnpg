MODULES := rnnpg rnnpg-decoder first-sentence-generator rnnpg-generator

debug :
	@for subdir in $(MODULES);\
		do $(MAKE) -C $$subdir DEBUGFLAGS=1;\
	done

release:
	@for subdir in $(MODULES);\
		do $(MAKE) -C $$subdir;\
	done

clean:
	@for subdir in $(MODULES);\
		do $(MAKE) -C $$subdir clean;\
	done
