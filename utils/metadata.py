coco_classes_pytorch = ['__background__',  # always index 0
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'N/A', 'stop_sign',
                'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball',
                'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
                'bottle', 'N/A', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'N/A', 'dining_table',
                'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush' ]

coco_classes = ['__background__',  # always index 0
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
                'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

hico_action_classes = ['adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with', 'buy', 'carry', 'catch',
                   'chase', 'check', 'clean', 'control', 'cook', 'cut', 'cut_with', 'direct', 'drag', 'dribble',
                   'drink_with', 'drive', 'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly',
                   'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 'hug', 'hunt', 'inspect',
                   'install', 'jump', 'kick', 'kiss', 'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light', 'load',
                  'lose', 'make', 'milk', 'move', 'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay',
                  'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race', 'read', 'release',
                   'repair', 'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign', 'sip', 'sit_at',
                   'sit_on', 'slide', 'smell', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 'stick', 'stir',
                   'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast',
                   'train', 'turn', 'type_on', 'walk', 'wash', 'watch', 'wave', 'wear', 'wield', 'zip']

hico_hoi_classes = ['board', 'direct', 'exit', 'fly', 'inspect', 'load', 'ride', 'sit_on', 'wash', 'no_interaction',
                'carry', 'hold', 'inspect', 'jump', 'hop_on', 'park', 'push', 'repair', 'ride', 'sit_on', 'straddle',
                'walk', 'wash', 'no_interaction', 'chase', 'feed', 'hold', 'pet', 'release', 'watch', 'no_interaction',
                'board', 'drive', 'exit', 'inspect', 'jump', 'launch', 'repair', 'ride', 'row', 'sail', 'sit_on',
                'stand_on', 'tie', 'wash', 'no_interaction', 'carry', 'drink_with', 'hold', 'inspect', 'lick', 'open',
               'pour', 'no_interaction', 'board', 'direct', 'drive', 'exit', 'inspect', 'load', 'ride', 'sit_on',
               'wash', 'wave', 'no_interaction', 'board', 'direct', 'drive', 'hose', 'inspect', 'jump', 'load', 'park',
               'ride', 'wash', 'no_interaction', 'dry', 'feed', 'hold', 'hug', 'kiss', 'pet', 'scratch', 'wash',
               'chase', 'no_interaction', 'carry', 'hold', 'lie_on', 'sit_on', 'stand_on', 'no_interaction', 'carry',
               'lie_on', 'sit_on', 'no_interaction', 'feed', 'herd', 'hold', 'hug', 'kiss', 'lasso', 'milk', 'pet',
               'ride', 'walk', 'no_interaction', 'clean', 'eat_at', 'sit_at', 'no_interaction', 'carry', 'dry', 'feed',
               'groom', 'hold', 'hose', 'hug', 'inspect', 'kiss', 'pet', 'run', 'scratch', 'straddle', 'train', 'walk',
               'wash', 'chase', 'no_interaction', 'feed', 'groom', 'hold', 'hug', 'jump', 'kiss', 'load', 'hop_on',
               'pet', 'race', 'ride', 'run', 'straddle', 'train', 'walk', 'wash', 'no_interaction', 'hold', 'inspect',
               'jump', 'hop_on', 'park', 'push', 'race', 'ride', 'sit_on', 'straddle', 'turn', 'walk', 'wash',
               'no_interaction', 'carry', 'greet', 'hold', 'hug', 'kiss', 'stab', 'tag', 'teach', 'lick',
               'no_interaction', 'carry', 'hold', 'hose', 'no_interaction', 'carry', 'feed', 'herd', 'hold', 'hug',
               'kiss', 'pet', 'ride', 'shear', 'walk', 'wash', 'no_interaction', 'board', 'drive', 'exit', 'load',
               'ride', 'sit_on', 'wash', 'no_interaction', 'control', 'repair', 'watch', 'no_interaction', 'buy',
               'cut', 'eat', 'hold', 'inspect', 'peel', 'pick', 'smell', 'wash', 'no_interaction', 'carry', 'hold',
               'inspect', 'open', 'wear', 'no_interaction', 'buy', 'carry', 'cut', 'eat', 'hold', 'inspect', 'peel',
               'pick', 'smell', 'no_interaction', 'break', 'carry', 'hold', 'sign', 'swing', 'throw', 'wield',
               'no_interaction', 'hold', 'wear', 'no_interaction', 'feed', 'hunt', 'watch', 'no_interaction', 'clean',
               'lie_on', 'sit_on', 'no_interaction', 'inspect', 'lie_on', 'sit_on', 'no_interaction', 'carry', 'hold',
               'open', 'read', 'no_interaction', 'hold', 'stir', 'wash', 'lick', 'no_interaction', 'cut', 'eat',
               'hold', 'smell', 'stir', 'wash', 'no_interaction', 'blow', 'carry', 'cut', 'eat', 'hold', 'light',
               'make', 'pick_up', 'no_interaction', 'carry', 'cook', 'cut', 'eat', 'hold', 'peel', 'smell', 'stir',
               'wash', 'no_interaction', 'carry', 'hold', 'read', 'repair', 'talk_on', 'text_on', 'no_interaction',
               'check', 'hold', 'repair', 'set', 'no_interaction', 'carry', 'drink_with', 'hold', 'inspect', 'pour',
               'sip', 'smell', 'fill', 'wash', 'no_interaction', 'buy', 'carry', 'eat', 'hold', 'make', 'pick_up',
               'smell', 'no_interaction', 'feed', 'hold', 'hose', 'hug', 'kiss', 'hop_on', 'pet', 'ride', 'walk',
               'wash', 'watch', 'no_interaction', 'hug', 'inspect', 'open', 'paint', 'no_interaction', 'hold', 'lift',
               'stick', 'lick', 'wash', 'no_interaction', 'block', 'catch', 'hold', 'spin', 'throw', 'no_interaction',
               'feed', 'kiss', 'pet', 'ride', 'watch', 'no_interaction', 'hold', 'operate', 'repair', 'no_interaction',
               'carry', 'hold', 'inspect', 'no_interaction', 'carry', 'cook', 'cut', 'eat', 'hold', 'make',
               'no_interaction', 'carry', 'clean', 'hold', 'type_on', 'no_interaction', 'assemble', 'carry', 'fly',
               'hold', 'inspect', 'launch', 'pull', 'no_interaction', 'cut_with', 'hold', 'stick', 'wash', 'wield',
               'lick', 'no_interaction', 'hold', 'open', 'read', 'repair', 'type_on', 'no_interaction', 'clean',
               'open', 'operate', 'no_interaction', 'control', 'hold', 'repair', 'no_interaction', 'buy', 'cut', 'eat',
               'hold', 'inspect', 'peel', 'pick', 'squeeze', 'wash', 'no_interaction', 'clean', 'hold', 'inspect',
               'open', 'repair', 'operate', 'no_interaction', 'check', 'pay', 'repair', 'no_interaction', 'buy',
               'carry', 'cook', 'cut', 'eat', 'hold', 'make', 'pick_up', 'slide', 'smell', 'no_interaction', 'clean',
               'hold', 'move', 'open', 'no_interaction', 'hold', 'point', 'swing', 'no_interaction', 'carry', 'cook',
               'cut', 'eat', 'hold', 'make', 'no_interaction', 'cut_with', 'hold', 'open', 'no_interaction', 'clean',
               'repair', 'wash', 'no_interaction', 'carry', 'flip', 'grind', 'hold', 'jump', 'pick_up', 'ride',
               'sit_on', 'stand_on', 'no_interaction', 'adjust', 'carry', 'hold', 'inspect', 'jump', 'pick_up',
               'repair', 'ride', 'stand_on', 'wear', 'no_interaction', 'adjust', 'carry', 'grind', 'hold', 'jump',
               'ride', 'stand_on', 'wear', 'no_interaction', 'hold', 'lick', 'wash', 'sip', 'no_interaction', 'block',
               'carry', 'catch', 'dribble', 'hit', 'hold', 'inspect', 'kick', 'pick_up', 'serve', 'sign', 'spin',
               'throw', 'no_interaction', 'hold', 'stand_under', 'stop_at', 'no_interaction', 'carry', 'drag', 'hold',
               'hug', 'load', 'open', 'pack', 'pick_up', 'zip', 'no_interaction', 'carry', 'drag', 'hold', 'inspect',
               'jump', 'lie_on', 'load', 'ride', 'stand_on', 'sit_on', 'wash', 'no_interaction', 'carry', 'hold',
               'hug', 'kiss', 'no_interaction', 'carry', 'hold', 'inspect', 'swing', 'no_interaction', 'adjust', 'cut',
               'hold', 'inspect', 'pull', 'tie', 'wear', 'no_interaction', 'hold', 'operate', 'repair',
               'no_interaction', 'clean', 'flush', 'open', 'repair', 'sit_on', 'stand_on', 'wash', 'no_interaction',
               'brush_with', 'hold', 'wash', 'no_interaction', 'install', 'repair', 'stand_under', 'stop_at',
               'no_interaction', 'direct', 'drive', 'inspect', 'load', 'repair', 'ride', 'sit_on', 'wash',
               'no_interaction', 'carry', 'hold', 'lose', 'open', 'repair', 'set', 'stand_under', 'no_interaction',
               'hold', 'make', 'paint', 'no_interaction', 'fill', 'hold', 'sip', 'toast', 'lick', 'wash',
               'no_interaction', 'feed', 'hold', 'pet', 'watch', 'no_interaction']

obj_hoi_index = [(0, 0), (161, 170), (11, 24), (66, 76), (147, 160), (1, 10), (55, 65), (187, 194), (568, 576),
                 (32, 46), (563, 567), (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), (77, 86), (112, 129),
                 (130, 146), (175, 186), (97, 107), (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), (577, 584),
                 (353, 356), (539, 546), (507, 516), (337, 342), (464, 474), (475, 483), (489, 502), (369, 376), (225, 232),
                 (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), (589, 595), (296, 305), (331, 336), (377, 383),
                 (484, 488), (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), (258, 264), (274, 283), (357, 363),
                 (419, 429), (306, 313), (265, 273), (87, 92), (93, 96), (171, 174), (240, 243), (108, 111), (551, 558),
                 (195, 198), (384, 389), (394, 397), (435, 438),(364, 368), (284, 290), (390, 393), (408, 414), (547, 550),
                 (450, 453), (430, 434), (248, 252), (291, 295),(585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
                ]

coco_pytorch_to_coco = ['N/A' if c not in coco_classes else coco_classes.index(c) for c in coco_classes_pytorch]
hoi_to_action = [hico_action_classes.index(c) for c in hico_hoi_classes]


# id   object          verb          
# -----------------------------------
# 001  airplane        board         
# 002  airplane        direct        
# 003  airplane        exit          
# 004  airplane        fly           
# 005  airplane        inspect       
# 006  airplane        load          
# 007  airplane        ride          
# 008  airplane        sit_on        
# 009  airplane        wash          
# 010  airplane        no_interaction
# 011  bicycle         carry         
# 012  bicycle         hold          
# 013  bicycle         inspect       
# 014  bicycle         jump          
# 015  bicycle         hop_on        
# 016  bicycle         park          
# 017  bicycle         push          
# 018  bicycle         repair        
# 019  bicycle         ride          
# 020  bicycle         sit_on        
# 021  bicycle         straddle      
# 022  bicycle         walk          
# 023  bicycle         wash          
# 024  bicycle         no_interaction
# 025  bird            chase         
# 026  bird            feed          
# 027  bird            hold          
# 028  bird            pet           
# 029  bird            release       
# 030  bird            watch         
# 031  bird            no_interaction
# 032  boat            board         
# 033  boat            drive         
# 034  boat            exit          
# 035  boat            inspect       
# 036  boat            jump          
# 037  boat            launch        
# 038  boat            repair        
# 039  boat            ride          
# 040  boat            row           
# 041  boat            sail          
# 042  boat            sit_on        
# 043  boat            stand_on      
# 044  boat            tie           
# 045  boat            wash          
# 046  boat            no_interaction
# 047  bottle          carry         
# 048  bottle          drink_with    
# 049  bottle          hold          
# 050  bottle          inspect       
# 051  bottle          lick          
# 052  bottle          open          
# 053  bottle          pour          
# 054  bottle          no_interaction
# 055  bus             board         
# 056  bus             direct        
# 057  bus             drive         
# 058  bus             exit          
# 059  bus             inspect       
# 060  bus             load          
# 061  bus             ride          
# 062  bus             sit_on        
# 063  bus             wash          
# 064  bus             wave          
# 065  bus             no_interaction
# 066  car             board         
# 067  car             direct        
# 068  car             drive         
# 069  car             hose          
# 070  car             inspect       
# 071  car             jump          
# 072  car             load          
# 073  car             park          
# 074  car             ride          
# 075  car             wash          
# 076  car             no_interaction
# 077  cat             dry           
# 078  cat             feed          
# 079  cat             hold          
# 080  cat             hug           
# 081  cat             kiss          
# 082  cat             pet           
# 083  cat             scratch       
# 084  cat             wash          
# 085  cat             chase         
# 086  cat             no_interaction
# 087  chair           carry         
# 088  chair           hold          
# 089  chair           lie_on        
# 090  chair           sit_on        
# 091  chair           stand_on      
# 092  chair           no_interaction
# 093  couch           carry         
# 094  couch           lie_on        
# 095  couch           sit_on        
# 096  couch           no_interaction
# 097  cow             feed          
# 098  cow             herd          
# 099  cow             hold          
# 100  cow             hug           
# 101  cow             kiss          
# 102  cow             lasso         
# 103  cow             milk          
# 104  cow             pet           
# 105  cow             ride          
# 106  cow             walk          
# 107  cow             no_interaction
# 108  dining_table    clean         
# 109  dining_table    eat_at        
# 110  dining_table    sit_at        
# 111  dining_table    no_interaction
# 112  dog             carry         
# 113  dog             dry           
# 114  dog             feed          
# 115  dog             groom         
# 116  dog             hold          
# 117  dog             hose          
# 118  dog             hug           
# 119  dog             inspect       
# 120  dog             kiss          
# 121  dog             pet           
# 122  dog             run           
# 123  dog             scratch       
# 124  dog             straddle      
# 125  dog             train         
# 126  dog             walk          
# 127  dog             wash          
# 128  dog             chase         
# 129  dog             no_interaction
# 130  horse           feed          
# 131  horse           groom         
# 132  horse           hold          
# 133  horse           hug           
# 134  horse           jump          
# 135  horse           kiss          
# 136  horse           load          
# 137  horse           hop_on        
# 138  horse           pet           
# 139  horse           race          
# 140  horse           ride          
# 141  horse           run           
# 142  horse           straddle      
# 143  horse           train         
# 144  horse           walk          
# 145  horse           wash          
# 146  horse           no_interaction
# 147  motorcycle      hold          
# 148  motorcycle      inspect       
# 149  motorcycle      jump          
# 150  motorcycle      hop_on        
# 151  motorcycle      park          
# 152  motorcycle      push          
# 153  motorcycle      race          
# 154  motorcycle      ride          
# 155  motorcycle      sit_on        
# 156  motorcycle      straddle      
# 157  motorcycle      turn          
# 158  motorcycle      walk          
# 159  motorcycle      wash          
# 160  motorcycle      no_interaction
# 161  person          carry         
# 162  person          greet         
# 163  person          hold          
# 164  person          hug           
# 165  person          kiss          
# 166  person          stab          
# 167  person          tag           
# 168  person          teach         
# 169  person          lick          
# 170  person          no_interaction
# 171  potted_plant    carry         
# 172  potted_plant    hold          
# 173  potted_plant    hose          
# 174  potted_plant    no_interaction
# 175  sheep           carry         
# 176  sheep           feed          
# 177  sheep           herd          
# 178  sheep           hold          
# 179  sheep           hug           
# 180  sheep           kiss          
# 181  sheep           pet           
# 182  sheep           ride          
# 183  sheep           shear         
# 184  sheep           walk          
# 185  sheep           wash          
# 186  sheep           no_interaction
# 187  train           board         
# 188  train           drive         
# 189  train           exit          
# 190  train           load          
# 191  train           ride          
# 192  train           sit_on        
# 193  train           wash          
# 194  train           no_interaction
# 195  tv              control       
# 196  tv              repair        
# 197  tv              watch         
# 198  tv              no_interaction
# 199  apple           buy           
# 200  apple           cut           
# 201  apple           eat           
# 202  apple           hold          
# 203  apple           inspect       
# 204  apple           peel          
# 205  apple           pick          
# 206  apple           smell         
# 207  apple           wash          
# 208  apple           no_interaction
# 209  backpack        carry         
# 210  backpack        hold          
# 211  backpack        inspect       
# 212  backpack        open          
# 213  backpack        wear          
# 214  backpack        no_interaction
# 215  banana          buy           
# 216  banana          carry         
# 217  banana          cut           
# 218  banana          eat           
# 219  banana          hold          
# 220  banana          inspect       
# 221  banana          peel          
# 222  banana          pick          
# 223  banana          smell         
# 224  banana          no_interaction
# 225  baseball_bat    break         
# 226  baseball_bat    carry         
# 227  baseball_bat    hold          
# 228  baseball_bat    sign          
# 229  baseball_bat    swing         
# 230  baseball_bat    throw         
# 231  baseball_bat    wield         
# 232  baseball_bat    no_interaction
# 233  baseball_glove  hold          
# 234  baseball_glove  wear          
# 235  baseball_glove  no_interaction
# 236  bear            feed          
# 237  bear            hunt          
# 238  bear            watch         
# 239  bear            no_interaction
# 240  bed             clean         
# 241  bed             lie_on        
# 242  bed             sit_on        
# 243  bed             no_interaction
# 244  bench           inspect       
# 245  bench           lie_on        
# 246  bench           sit_on        
# 247  bench           no_interaction
# 248  book            carry         
# 249  book            hold          
# 250  book            open          
# 251  book            read          
# 252  book            no_interaction
# 253  bowl            hold          
# 254  bowl            stir          
# 255  bowl            wash          
# 256  bowl            lick          
# 257  bowl            no_interaction
# 258  broccoli        cut           
# 259  broccoli        eat           
# 260  broccoli        hold          
# 261  broccoli        smell         
# 262  broccoli        stir          
# 263  broccoli        wash          
# 264  broccoli        no_interaction
# 265  cake            blow          
# 266  cake            carry         
# 267  cake            cut           
# 268  cake            eat           
# 269  cake            hold          
# 270  cake            light         
# 271  cake            make          
# 272  cake            pick_up       
# 273  cake            no_interaction
# 274  carrot          carry         
# 275  carrot          cook          
# 276  carrot          cut           
# 277  carrot          eat           
# 278  carrot          hold          
# 279  carrot          peel          
# 280  carrot          smell         
# 281  carrot          stir          
# 282  carrot          wash          
# 283  carrot          no_interaction
# 284  cell_phone      carry         
# 285  cell_phone      hold          
# 286  cell_phone      read          
# 287  cell_phone      repair        
# 288  cell_phone      talk_on       
# 289  cell_phone      text_on       
# 290  cell_phone      no_interaction
# 291  clock           check         
# 292  clock           hold          
# 293  clock           repair        
# 294  clock           set           
# 295  clock           no_interaction
# 296  cup             carry         
# 297  cup             drink_with    
# 298  cup             hold          
# 299  cup             inspect       
# 300  cup             pour          
# 301  cup             sip           
# 302  cup             smell         
# 303  cup             fill          
# 304  cup             wash          
# 305  cup             no_interaction
# 306  donut           buy           
# 307  donut           carry         
# 308  donut           eat           
# 309  donut           hold          
# 310  donut           make          
# 311  donut           pick_up       
# 312  donut           smell         
# 313  donut           no_interaction
# 314  elephant        feed          
# 315  elephant        hold          
# 316  elephant        hose          
# 317  elephant        hug           
# 318  elephant        kiss          
# 319  elephant        hop_on        
# 320  elephant        pet           
# 321  elephant        ride          
# 322  elephant        walk          
# 323  elephant        wash          
# 324  elephant        watch         
# 325  elephant        no_interaction
# 326  fire_hydrant    hug           
# 327  fire_hydrant    inspect       
# 328  fire_hydrant    open          
# 329  fire_hydrant    paint         
# 330  fire_hydrant    no_interaction
# 331  fork            hold          
# 332  fork            lift          
# 333  fork            stick         
# 334  fork            lick          
# 335  fork            wash          
# 336  fork            no_interaction
# 337  frisbee         block         
# 338  frisbee         catch         
# 339  frisbee         hold          
# 340  frisbee         spin          
# 341  frisbee         throw         
# 342  frisbee         no_interaction
# 343  giraffe         feed          
# 344  giraffe         kiss          
# 345  giraffe         pet           
# 346  giraffe         ride          
# 347  giraffe         watch         
# 348  giraffe         no_interaction
# 349  hair_drier      hold          
# 350  hair_drier      operate       
# 351  hair_drier      repair        
# 352  hair_drier      no_interaction
# 353  handbag         carry         
# 354  handbag         hold          
# 355  handbag         inspect       
# 356  handbag         no_interaction
# 357  hot_dog         carry         
# 358  hot_dog         cook          
# 359  hot_dog         cut           
# 360  hot_dog         eat           
# 361  hot_dog         hold          
# 362  hot_dog         make          
# 363  hot_dog         no_interaction
# 364  keyboard        carry         
# 365  keyboard        clean         
# 366  keyboard        hold          
# 367  keyboard        type_on       
# 368  keyboard        no_interaction
# 369  kite            assemble      
# 370  kite            carry         
# 371  kite            fly           
# 372  kite            hold          
# 373  kite            inspect       
# 374  kite            launch        
# 375  kite            pull          
# 376  kite            no_interaction
# 377  knife           cut_with      
# 378  knife           hold          
# 379  knife           stick         
# 380  knife           wash          
# 381  knife           wield         
# 382  knife           lick          
# 383  knife           no_interaction
# 384  laptop          hold          
# 385  laptop          open          
# 386  laptop          read          
# 387  laptop          repair        
# 388  laptop          type_on       
# 389  laptop          no_interaction
# 390  microwave       clean         
# 391  microwave       open          
# 392  microwave       operate       
# 393  microwave       no_interaction
# 394  mouse           control       
# 395  mouse           hold          
# 396  mouse           repair        
# 397  mouse           no_interaction
# 398  orange          buy           
# 399  orange          cut           
# 400  orange          eat           
# 401  orange          hold          
# 402  orange          inspect       
# 403  orange          peel          
# 404  orange          pick          
# 405  orange          squeeze       
# 406  orange          wash          
# 407  orange          no_interaction
# 408  oven            clean         
# 409  oven            hold          
# 410  oven            inspect       
# 411  oven            open          
# 412  oven            repair        
# 413  oven            operate       
# 414  oven            no_interaction
# 415  parking_meter   check         
# 416  parking_meter   pay           
# 417  parking_meter   repair        
# 418  parking_meter   no_interaction
# 419  pizza           buy           
# 420  pizza           carry         
# 421  pizza           cook          
# 422  pizza           cut           
# 423  pizza           eat           
# 424  pizza           hold          
# 425  pizza           make          
# 426  pizza           pick_up       
# 427  pizza           slide         
# 428  pizza           smell         
# 429  pizza           no_interaction
# 430  refrigerator    clean         
# 431  refrigerator    hold          
# 432  refrigerator    move          
# 433  refrigerator    open          
# 434  refrigerator    no_interaction
# 435  remote          hold          
# 436  remote          point         
# 437  remote          swing         
# 438  remote          no_interaction
# 439  sandwich        carry         
# 440  sandwich        cook          
# 441  sandwich        cut           
# 442  sandwich        eat           
# 443  sandwich        hold          
# 444  sandwich        make          
# 445  sandwich        no_interaction
# 446  scissors        cut_with      
# 447  scissors        hold          
# 448  scissors        open          
# 449  scissors        no_interaction
# 450  sink            clean         
# 451  sink            repair        
# 452  sink            wash          
# 453  sink            no_interaction
# 454  skateboard      carry         
# 455  skateboard      flip          
# 456  skateboard      grind         
# 457  skateboard      hold          
# 458  skateboard      jump          
# 459  skateboard      pick_up       
# 460  skateboard      ride          
# 461  skateboard      sit_on        
# 462  skateboard      stand_on      
# 463  skateboard      no_interaction
# 464  skis            adjust        
# 465  skis            carry         
# 466  skis            hold          
# 467  skis            inspect       
# 468  skis            jump          
# 469  skis            pick_up       
# 470  skis            repair        
# 471  skis            ride          
# 472  skis            stand_on      
# 473  skis            wear          
# 474  skis            no_interaction
# 475  snowboard       adjust        
# 476  snowboard       carry         
# 477  snowboard       grind         
# 478  snowboard       hold          
# 479  snowboard       jump          
# 480  snowboard       ride          
# 481  snowboard       stand_on      
# 482  snowboard       wear          
# 483  snowboard       no_interaction
# 484  spoon           hold          
# 485  spoon           lick          
# 486  spoon           wash          
# 487  spoon           sip           
# 488  spoon           no_interaction
# 489  sports_ball     block         
# 490  sports_ball     carry         
# 491  sports_ball     catch         
# 492  sports_ball     dribble       
# 493  sports_ball     hit           
# 494  sports_ball     hold          
# 495  sports_ball     inspect       
# 496  sports_ball     kick          
# 497  sports_ball     pick_up       
# 498  sports_ball     serve         
# 499  sports_ball     sign          
# 500  sports_ball     spin          
# 501  sports_ball     throw         
# 502  sports_ball     no_interaction
# 503  stop_sign       hold          
# 504  stop_sign       stand_under   
# 505  stop_sign       stop_at       
# 506  stop_sign       no_interaction
# 507  suitcase        carry         
# 508  suitcase        drag          
# 509  suitcase        hold          
# 510  suitcase        hug           
# 511  suitcase        load          
# 512  suitcase        open          
# 513  suitcase        pack          
# 514  suitcase        pick_up       
# 515  suitcase        zip           
# 516  suitcase        no_interaction
# 517  surfboard       carry         
# 518  surfboard       drag          
# 519  surfboard       hold          
# 520  surfboard       inspect       
# 521  surfboard       jump          
# 522  surfboard       lie_on        
# 523  surfboard       load          
# 524  surfboard       ride          
# 525  surfboard       stand_on      
# 526  surfboard       sit_on        
# 527  surfboard       wash          
# 528  surfboard       no_interaction
# 529  teddy_bear      carry         
# 530  teddy_bear      hold          
# 531  teddy_bear      hug           
# 532  teddy_bear      kiss          
# 533  teddy_bear      no_interaction
# 534  tennis_racket   carry         
# 535  tennis_racket   hold          
# 536  tennis_racket   inspect       
# 537  tennis_racket   swing         
# 538  tennis_racket   no_interaction
# 539  tie             adjust        
# 540  tie             cut           
# 541  tie             hold          
# 542  tie             inspect       
# 543  tie             pull          
# 544  tie             tie           
# 545  tie             wear          
# 546  tie             no_interaction
# 547  toaster         hold          
# 548  toaster         operate       
# 549  toaster         repair        
# 550  toaster         no_interaction
# 551  toilet          clean         
# 552  toilet          flush         
# 553  toilet          open          
# 554  toilet          repair        
# 555  toilet          sit_on        
# 556  toilet          stand_on      
# 557  toilet          wash          
# 558  toilet          no_interaction
# 559  toothbrush      brush_with    
# 560  toothbrush      hold          
# 561  toothbrush      wash          
# 562  toothbrush      no_interaction
# 563  traffic_light   install       
# 564  traffic_light   repair        
# 565  traffic_light   stand_under   
# 566  traffic_light   stop_at       
# 567  traffic_light   no_interaction
# 568  truck           direct        
# 569  truck           drive         
# 570  truck           inspect       
# 571  truck           load          
# 572  truck           repair        
# 573  truck           ride          
# 574  truck           sit_on        
# 575  truck           wash          
# 576  truck           no_interaction
# 577  umbrella        carry         
# 578  umbrella        hold          
# 579  umbrella        lose          
# 580  umbrella        open          
# 581  umbrella        repair        
# 582  umbrella        set           
# 583  umbrella        stand_under   
# 584  umbrella        no_interaction
# 585  vase            hold          
# 586  vase            make          
# 587  vase            paint         
# 588  vase            no_interaction
# 589  wine_glass      fill          
# 590  wine_glass      hold          
# 591  wine_glass      sip           
# 592  wine_glass      toast         
# 593  wine_glass      lick          
# 594  wine_glass      wash          
# 595  wine_glass      no_interaction
# 596  zebra           feed          
# 597  zebra           hold          
# 598  zebra           pet           
# 599  zebra           watch         
# 600  zebra           no_interaction