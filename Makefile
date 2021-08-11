
GIT_HOOKS := .git/hooks/installed

all : $(GIT_HOOKS)

$(GIT_HOOKS):
	@scripts/hooks_installation.sh
	@echo

