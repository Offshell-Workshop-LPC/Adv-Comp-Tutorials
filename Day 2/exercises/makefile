COMPILEPATH          = ./
BINDIR               = $(COMPILEPATH)/bin/
EXEDIR               = $(COMPILEPATH)/executables/

CXX           = nvcc
CXXVEROPT     = -std=c++20
CXXOPTIM      = -O2
CXXDEFINES    =
NVLINKOPTS    = -Xnvlink --suppress-stack-size-warning
CXXFLAGS      = -dc -rdc=true -x cu --cudart=shared $(NVLINKOPTS) -Xcompiler -fPIC -g $(CXXOPTIM) $(CXXVEROPT)
EXEFLAGS      = $(filter-out -dc, $(CXXFLAGS))

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
