
GIT_HOOKS := .git/hooks/installed
scripts := scripts/commit-msg.sh scripts/hooks_installation.sh scripts/pre-commit.sh

all : $(GIT_HOOKS)

$(GIT_HOOKS): $(scripts)
	@scripts/hooks_installation.sh
	@echo

clean:
	-rm .git/hooks/installed

