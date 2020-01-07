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
"set softtabstop=4
"set expandtab
set shiftwidth=4
set showcmd		" Show (partial) command in status line.
set showmatch		" Show matching brackets.
set ignorecase		" Do case insensitive matching
set smartcase		" Do smart case matching
set incsearch		" Incremental search
set autowrite		" Automatically save before commands like :next and :make
set hidden		" Hide buffers when they are abandoned
set mouse=a		" Enable mouse usage (all modes)
"colorscheme desert	" elflord ron peachpuff default
"set background=dark 
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
    

call plug#begin()
Plug 'roxma/nvim-completion-manager'
Plug 'SirVer/ultisnips'
Plug 'honza/vim-snippets'
Plug 'scrooloose/nerdcommenter'
Plug 'sbdchd/neoformat'
"Plug 'davidhalter/jedi-vim'
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'roxma/ncm-clang'
"Plug 'Shougo/deoplete.nvim'
Plug 'zchee/deoplete-jedi'
Plug 'jiangmiao/auto-pairs'
Plug 'scrooloose/nerdtree'
Plug 'tmhedberg/SimpylFold'
Plug 'morhetz/gruvbox'
Plug 'Shougo/deoplete-clangx'
call plug#end()
"colorscheme gruvbox
" disable autocompletion, cause we use deoplete for completion
let g:jedi#completions_enabled = 0

" open the go-to function in split, not another buffer
"let g:jedi#use_splits_not_buffers = "right"
"let g:deoplete#enable_at_startup = 1
let g:SimpylFold_docstring_preview = 1
inoremap <expr><tab> pumvisible() ? "\<c-n>" : "\<tab>"


if has('nvim')
  Plug 'Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
else
  Plug 'Shougo/deoplete.nvim'
  Plug 'roxma/nvim-yarp'
  Plug 'roxma/vim-hug-neovim-rpc'
endif

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

"noremap <C-b> :!python %<cr>
autocmd FileType python noremap <C-b> :!python %<cr>
autocmd FileType cpp  noremap <C-b> :!g++ %  -std=c++11 -lpthread `pkg-config --libs --cflags opencv` -g -o %-build && ./%-build<cr>
autocmd FileType c  noremap <C-b> :!gcc % -g -o %-build && ./%-build<cr>
autocmd FileType sh  noremap <C-b> :!./%<cr>
autocmd FileType java  noremap <C-b> :!javac % && java %:r<cr>

