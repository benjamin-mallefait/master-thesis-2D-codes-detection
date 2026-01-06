# Makefile for Thesis Project (Mac/Linux only)

VENV_NAME = venv
PYTHON = python3

# Crée ou met à jour l'environnement avec virtualenv
venv:
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "Environnement virtuel absent, création avec virtualenv..."; \
		if ! command -v virtualenv >/dev/null 2>&1; then \
			echo "'virtualenv' n'est pas installé. Installe-le avec : pip install --user virtualenv"; \
			exit 1; \
		fi; \
		virtualenv $(VENV_NAME); \
	else \
		echo "Environnement virtuel déjà présent."; \
	fi
	echo "Activation de l'environnement virtuel et installation des dépendances... (peut prendre un moment)"
	@. $(VENV_NAME)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

venv_tmux:
	@tmux has-session -t install_venv 2>/dev/null || \
	tmux new-session -d -s install_venv 'make venv'

# Supprimer et recréer entièrement l'environnement
.PHONY: reset
reset:
	rm -rf $(VENV_NAME)
	$(MAKE) venv

# Affiche la commande pour activer le venv
activate:
	echo "source $(VENV_NAME)/bin/activate"

# Installe les dépendances si l'env est déjà activé
update:
	pip install -r requirements.txt

# Enregistre les paquets dans requirements.txt
freeze:
	. $(VENV_NAME)/bin/activate && pip freeze > requirements.txt

pull-results:
	@. $(VENV_NAME)/bin/activate && scripts/pull_results.sh $(SSH_CONFIG)