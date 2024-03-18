COMPILEPATH          = ./
BINDIR               = $(COMPILEPATH)/bin/
EXEDIR               = $(COMPILEPATH)/executables/

CXX           = g++
CXXVEROPT     = -std=c++20
CXXOPTIM      = -O2
CXXDEFINES    =
CXXFLAGS      = -fPIC -g $(CXXOPTIM) $(CXXVEROPT)
EXEFLAGS      = $(CXXFLAGS)

BINSCC = $(wildcard $(BINDIR)*.cc)
BINSCXX = $(wildcard $(BINDIR)*.cxx)
EXESPRIM = $(BINSCC:.cc=) $(BINSCXX:.cxx=)
EXES = $(subst $(BINDIR),$(EXEDIR),$(EXESPRIM))


.PHONY: all help clean
.SILENT: alldirs clean $(EXES)


all:	$(EXES)


alldirs:
	mkdir -p $(EXEDIR)


$(EXEDIR)%::	$(BINDIR)%.cc | alldirs
	echo "Compiling $<"; \
	$(CXX) $(EXEFLAGS) -o $@ $< $(LIBS)


clean:
	rm -rf $(EXEDIR)
	rm -f $(BINDIR)*.o
	rm -f $(BINDIR)*.so
	rm -f $(BINDIR)*.d


include $(DEPS)
