#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   chunker.py                                           :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/22 14:51:11 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/25 16:52:11 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

# 3 chunker different, 1 pour les markdown, code python, doc

import ast  # noqa
import os
import mimetypes


from typing import Tuple

Chunk = Tuple[str, int, int, str]


def c_markdown() -> None:
    path = "./student/src/fokepasf.txt"
    os.chmod(path, 0o0700)
    print(mimetypes.guess_type(path))
    # print(os.system(path))


def chunker(file_path: str, content: str, max_size: int) -> list[Chunk]:
    pass


c_markdown()

# tree = ast.parse("x = 1 + 2")
# print("\n", tree)
# tree = ast.dump(tree, indent=4)
# print("\n", tree)
