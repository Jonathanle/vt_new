" Plugin Manager from Vim

"
"set rtp+=~/.vim/bundle/Vundle.vim
"call vundle#begin()
"
"Plugin 'VundleVim/Vundle.vim'
"
"Plugin 'jupyter-vim/jupyter-vim'
"
"
"set nocompatible              " be iMproved, required
"filetype off                  " required
"filetype plugin indent on    " required
"
"call vundle#end()            " required
"filetype plugin indent on    " required
"
" End Vundle

set foldmethod=manual

" made so that I can create the directory *if* it does not exist
if !isdirectory($HOME.'/.vim/view')
    silent call mkdir($HOME.'/.vim/view', 'p')
endif
" Set the view directory
set viewdir=~/.vim/view

" Commands to load and remove the views
autocmd BufWinLeave *.* mkview
autocmd BufWinEnter *.* silent loadview

syntax on
set number

" Pytest Command to Load the functions
noremap q- :!pytest<CR>


set tabstop=4
set shiftwidth=4
set expandtab
set softtabstop=4
