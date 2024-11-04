    # Obtém o diretório do arquivo atual
    current_dir = os.path.dirname(__file__)
    
    # Navega até o diretório pai do pai (ou seja, dois níveis acima)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # Define o diretório pai do pai como diretório atual
    os.chdir(parent_dir)
    
    # Opcional: Adiciona o diretório pai do pai ao sys.path para permitir importações
    sys.path.append(parent_dir)

    print("Diretório definido:", parent_dir)
