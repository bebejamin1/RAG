#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 13:18:12 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/26 11:18:06 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import colorama as c
import os
import fire

from student.src.indexing import _PROJECT_ROOT, index_main


# *****************************************************************************
# *                               CLASS                                       *
# *                                                                           *

class RagSystem():

    def __init__(self):
        self.ra = c.Style.RESET_ALL
        self.rs = "\033[0m"
        self.r = "\033[31m\033[5m\033[1m"

# ============================ LOAD_INDEX =====================================

    def index(self,
              repo_path: str = f"{_PROJECT_ROOT}/data/raw/vllm-0.10.1",
              max_chunk_size: int = 2000) -> None:

        os.system("clear")
        print("\n" + c.Fore.CYAN + "".center(79, "="))
        print(" INDEXING ".center(79, "="))
        print("".center(79, "=") + self.ra + "\n\n")

        try:

            index_main(repo_path, max_chunk_size)

        except Exception as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit()


# *****************************************************************************
# *                                MAIN                                       *
# *                                                                           *

def main():
    fire.Fire(RagSystem)


if __name__ == "__main__":
    main()
