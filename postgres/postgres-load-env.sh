#!/bin/bash

WD=$(pwd)

if [ -z "$1" ] ; then
	PG_INSTALL_DIR=$WD/postgres-server
elif [[ "$1" = /* ]] ; then
	PG_INSTALL_DIR="$1"
else
	PG_INSTALL_DIR="$WD/$1"
fi

cd $PG_INSTALL_DIR

PG_BIN_PATH="$PG_INSTALL_DIR/build/bin"
INIT=$(echo "$PATH" | grep "$PG_BIN_PATH")

if [ -z "$INIT" ] ; then
	export PG_BIN_PATH
	export PG_CTL_PATH="$WD"
	export PATH="$PG_BIN_PATH:$PATH"
	export LD_LIBRARY_PATH="$PG_INSTALL_DIR/build/lib:$LD_LIBRARY_PATH"
fi

cd $WD
