" Vim5 and later versions support syntax highlighting. Uncommenting the next
" line enables syntax highlighting by default.
"if has("syntax")
"  syntax on
"endif

" Uncomment the next line to make Vim more Vi-compatible
" NOTE: debian.vim sets 'nocompatible'. Setting 'compatible' changes numerous
" options, so any other options should be set AFTER setting 'compatible'.
set nocompatible
set tabstop=4
set softtabstop=4
set expandtab
set shiftwidth=4
set showcmd		" Show (partial) command in status line.
set showmatch		" Show matching brackets.
set ignorecase		" Do case insensitive matching
set smartcase		" Do smart case matching
set incsearch		" Incremental search
set autowrite		" Automatically save before commands like :next and :make
set hidden		" Hide buffers when they are abandoned
set mouse=a		" Enable mouse usage (all modes)
colorscheme desert	" elflord ron peachpuff default
set background=dark 
set autoindent        
set smartindent            
set backspace=2    " 设置退格键可用 
set linebreak        " 整词换行
set whichwrap=b,s,<,>,[,] " 光标从行首和行末时可以跳到另一行去
set number            " Enable line number    "显示行号
set history=50        " set command history to 50    "历史记录50条
set laststatus=2 " 总显示最后一个窗口的状态行；设为1则窗口数多于一个的时候显示最后一个窗口的状态行；0不显示最后一个窗口的状态行
set ruler            " 标尺，用于显示光标位置的行号和列号，逗号分隔。每个窗口都有自己的标尺。如果窗口有状态行，标尺在那里显示。否则，它显示在屏幕的最后一行上。
set showmode        " 命令行显示vim当前模式
set incsearch        " 输入字符串就显示匹配点
set hlsearch 
set fileencodings=utf-8,gbk,big5,cp936,gb18030,gb2312,utf-16
set guifont=Monospace\ 13  
    

" detect file type
    filetype on
    set nocp
	execute pathogen#infect()
    syntax on
    filetype plugin indent on

"-- Taglist setting --
    let Tlist_Ctags_Cmd='ctags' "因为我们放在环境变量里，所以可以直接执行
    let Tlist_Use_Right_Window=1 "让窗口显示在右边，0的话就是显示在左边
    let Tlist_Show_One_File=0 "让taglist可以同时展示多个文件的函数列表
    let Tlist_File_Fold_Auto_Close=1 "非当前文件，函数列表折叠隐藏
    let Tlist_Exit_OnlyWindow=1 "当taglist是最后一个分割窗口时，自动推出vim
    "是否一直处理tags.1:处理;0:不处理
    let Tlist_Process_File_Always=1 "实时更新tags
    let Tlist_Inc_Winwidth=0




"使用CTRL+[hjkl]在窗口间导航  
"map <C-c> <C-W>c  
map <C-j> <C-W>j  
map <C-k> <C-W>k  
map <C-h> <C-W>h  
map <C-l> <C-W>l  
"map <C-c> <C-W>c 

"括号匹配  
vnoremap $1 <esc>`>a)<esc>`<i(<esc>  
vnoremap $2 <esc>`>a]<esc>`<i[<esc>  
vnoremap $3 <esc>`>a}<esc>`<i{<esc>  
vnoremap $$ <esc>`>a"<esc>`<i"<esc>  
vnoremap $q <esc>`>a'<esc>`<i'<esc>  
vnoremap $e <esc>`>a"<esc>`<i"<esc>  
"非常好用的括号匹配，实际是自动生成括号  

noremap <C-b> :!python %<cr>


filetype plugin on
set dictionary=/usr/share/dict/complete-dict
let g:pydiction_menu_height = 4
filetype plugin indent on    
set completeopt=longest,menu
set complete-=k complete+=k
let g:pydiction_location = '/home/lhc/.vim/tools/pydiction/complete-dict'
map <C-t> :NERDTreeToggle<CR>
"autocmd vimenter * NERDTree

let g:Powerline_symbols = 'fancy'
let NERDTreeCascadeSingleChildDir=0
set fencs=utf-8,gbk,big5,cp936,gb18030,gb2312,utf-16
set laststatus=2
set t_Co=256   

let g:NERDTreeDirArrowExpandable = '+'
let g:NERDTreeDirArrowCollapsible = '-'

let mapleader='\'
let g:EasyMotion_do_mapping = 0 " Disable default mappings
map <Leader><Leader>f <Plug>(easymotion-bd-f)
"map<Leader><Leader>F<Plug>(easymotion-overwin-f2)

" Turn on case insensitive feature
let g:EasyMotion_smartcase = 1

" JK motions: Line motions
map <Leader><Leader>j <Plug>(easymotion-j)
map <Leader><Leader>k <Plug>(easymotion-k)
" Move to word
map  <Leader><Leader>w <Plug>(easymotion-bd-w)

" Move to line
map <Leader><Leader>l <Plug>(easymotion-bd-jk)
