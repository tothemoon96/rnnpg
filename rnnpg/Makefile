# change the target if you need
OUT_EXEC := rnnpg
OBJS := $(patsubst %.cpp, %.o, $(wildcard *.cpp))

CC := g++
CPPFLAGS := -O3 -funroll-loops -ffast-math -finline-functions -Wall -Winline -pipe -DKENLM_MAX_ORDER=6

ifdef DEBUGFLAGS
	ifeq ($(DEBUGFLAGS),1)
		ifeq "$(origin DEBUGFLAGS)" "command line"
			CPPFLAGS := -Og -g3 -Wall -Winline -pipe -DKENLM_MAX_ORDER=6
		endif
	endif
endif

INCLUDES := -I$(KENLM_PATH)
LDFLAGS := -L$(KENLM_PATH)/lib -lkenlm

all : $(OUT_EXEC)
#	rm *.o

$(OUT_EXEC) : $(OBJS)
	$(CC) -o $@ $^ $(INCLUDES) $(LDFLAGS)

%.o : %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@ $(INCLUDES)

clean:
	rm -f *.o
	rm -f $(OUT_EXEC)
