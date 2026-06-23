#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   chunker.py                                           :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/22 14:51:11 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/23 09:55:55 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

# 3 chunker different, 1 pour les markdown, code python, doc

import ast  # noqa

from typing import Tuple

Chunk = Tuple[str, int, int, str]


def chunker() -> None:
    pass
