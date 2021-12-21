/* A Bison parser, made by GNU Bison 3.7.5.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_NCG_NCGEN_TAB_H_INCLUDED
# define YY_NCG_NCGEN_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int ncgdebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    NC_UNLIMITED_K = 258,          /* NC_UNLIMITED_K  */
    BYTE_K = 259,                  /* BYTE_K  */
    CHAR_K = 260,                  /* CHAR_K  */
    SHORT_K = 261,                 /* SHORT_K  */
    INT_K = 262,                   /* INT_K  */
    FLOAT_K = 263,                 /* FLOAT_K  */
    DOUBLE_K = 264,                /* DOUBLE_K  */
    IDENT = 265,                   /* IDENT  */
    TERMSTRING = 266,              /* TERMSTRING  */
    BYTE_CONST = 267,              /* BYTE_CONST  */
    CHAR_CONST = 268,              /* CHAR_CONST  */
    SHORT_CONST = 269,             /* SHORT_CONST  */
    INT_CONST = 270,               /* INT_CONST  */
    FLOAT_CONST = 271,             /* FLOAT_CONST  */
    DOUBLE_CONST = 272,            /* DOUBLE_CONST  */
    DIMENSIONS = 273,              /* DIMENSIONS  */
    VARIABLES = 274,               /* VARIABLES  */
    NETCDF = 275,                  /* NETCDF  */
    DATA = 276,                    /* DATA  */
    FILLVALUE = 277                /* FILLVALUE  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE ncglval;

int ncgparse (void);

#endif /* !YY_NCG_NCGEN_TAB_H_INCLUDED  */
