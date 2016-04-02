// -*- c++ -*-

// guylib library - some useful tools you might enjoy
//
// Copyright (C) 2011-2016 Guy Bensky
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see
// <http://www.gnu.org/licenses/>.
//
// For more information, bug reports, or to contribute email me at:
// guylib@rabensky.com
//
// or you can look at the git repository at
// https://bitbucket.org/guyben/guylib

#include "guylib.h"
#include <algorithm>
#include <glob.h>

namespace guylib{

	bool assertClass::_throw=false;
	std::ostream *assertClass::deb_out=&std::cerr;

	void assertClass::apply(){
		first=false;
		std::string str=_out.str();
		while (str.size() && str[str.size()-1]=='\n'){
			str.erase(str.end()-1);
		}
		if (str.find('\n')!=std::string::npos){
			str="===============\n"+str+"\n===============";
		}
		if (_exit && _throw){
			throw assertException(str);
		}else{
			if (deb_out)
				(*deb_out)<<str<<std::endl;
			if (_exit)
				exit(-1);
		}
	}


	sstate::sstate(){
			clear();
		}
	sstate::sstate(const std::string &pat){
			assertErr(from(pat))<<"pat= "<<pat;
		}
	sstate::sstate(const char *pat){
			assertErr(from(pat))<<"pat= "<<pat;
		}
	sstate::sstate(const std::ostream &s){
			from(s);
		}
		void sstate::clear(){
			width=0;
			precision=6;
			fill=' ';
			flags=std::ios::dec | std::ios::skipws;
		}
		bool sstate::from(const char *pat){
			sstate tmp;
			tmp.flags^=tmp.flags;
			char *e;
			for(;*pat;++pat){
				switch(*pat){
					case ' ':
						continue;
					case '0':
						tmp.fill='0';
						continue;
					case '-':
						++pat;
						if (!*pat) return false;
						tmp.fill=*pat;
						continue;
					case '+':
						tmp.flags |= std::ios::showpos;
						continue;
					case 'p':
					case 'P':
						tmp.flags |= std::ios::showpoint;
						continue;
					case '.':
						++pat;
						if (!*pat) return false;
						tmp.precision=std::strtol(pat,&e,10);
						if (pat==e) return false;
						pat=e-1;
						if (tmp.precision<0) return false;
						continue;
					case '1':
					case '2':
					case '3':
					case '4':
					case '5':
					case '6':
					case '7':
					case '8':
					case '9':
						tmp.width=std::strtol(pat,&e,10);
						if (pat==e) return false;
						pat=e-1;
						if (tmp.width<0) return false;
						continue;
					case 'X':
						tmp.flags|=std::ios::showbase;
					case 'x':
						tmp.flags|=std::ios::hex;
						continue;
					case 'D':
						tmp.flags|=std::ios::showbase;
					case 'd':
						tmp.flags|=std::ios::dec;
						continue;
					case 'O':
						tmp.flags|=std::ios::showbase;
					case 'o':
						tmp.flags|=std::ios::oct;
						continue;
					case 'B':
					case 'b':
						tmp.flags|=std::ios::boolalpha;
						continue;
					case 'U':
					case 'u':
						tmp.flags|=std::ios::uppercase;
						continue;
					case 'F':
					case 'f':
						tmp.flags|=std::ios::fixed;
						continue;
					case 'S':
					case 's':
						tmp.flags|=std::ios::scientific;
						continue;
					case '<':
						tmp.flags|=std::ios::left;
						continue;
					case '>':
						tmp.flags|=std::ios::right;
						continue;
					case '=':
						tmp.flags|=std::ios::internal;
						continue;
					default:
						return false;
				}
			}
			*this=tmp;
			return true;
		}
		void sstate::from(const std::ios &s){
			precision=s.precision();
			width=s.width();
			fill=s.fill();
			flags=s.flags();
		}
		void sstate::to(std::ios &s)const{
			s.flags(flags);
			s.width(width);
			s.precision(precision);
			s.fill(fill);
		}
		std::string sstate::str()const{
			std::ostringstream res;
			if (flags&std::ios::showpos)res<<'+';
			if (fill=='0')
				res<<'0';
			if (width!=0)
				res<<width;
			res<<'.'<<precision;
			if (flags&std::ios::showpoint)res<<'p';
			if (flags&std::ios::hex)res<<((flags&std::ios::showbase)?'X':'x');
			if (flags&std::ios::oct)res<<((flags&std::ios::showbase)?'O':'o');
			if (flags&std::ios::dec)res<<((flags&std::ios::showbase)?'D':'d');
			if (flags&std::ios::boolalpha)res<<'b';
			if (flags&std::ios::uppercase)res<<'u';
			if (flags&std::ios::fixed)res<<'f';
			if (flags&std::ios::scientific)res<<'s';
			if (flags&std::ios::left) res<<'<';
			if (flags&std::ios::right) res<<'>';
			if (flags&std::ios::internal) res<<'=';
			if (fill!='0' && fill!=' ')
				res<<'-'<<fill;
			return res.str();
		}

	void add_encoded_char(std::ostream &out, char c, char sep){
		if (std::isprint(c) && c!='\\'){
			if (c==sep)
				out.put('\\');
			out.put(c);
			return;
		}
		switch(c){
			case '\\':
				out.put('\\');
				out.put('\\');
				return;
			case '\n':
				out.put('\\');
				out.put('n');
				return;
			case '\r':
				out.put('\\');
				out.put('r');
				return;
			case '\t':
				out.put('\\');
				out.put('t');
				return;
			case '\0':
				out.put('\\');
				out.put('0');
				return;
			default:
				out.put('\\');
				out.put('x');
				out.put("0123456789ABCDEF"[((int)(unsigned char)c)>>4]);
				out.put("0123456789ABCDEF"[((int)(unsigned char)c)&0xF]);
				return;
		}
	}

	void encode_into(std::ostream &out, const std::string &str, char sep){
		for (size_t i=0;i<str.size();++i)
			add_encoded_char(out,str[i],sep);
	}

	std::string encode(const std::string &str, char sep){
		std::ostringstream res;
		encode_into(res,str,sep);
		return res.str();
	}

	int hex2num(char c){
		if (c>='0' && c<='9')
			return int(c-'0');
		if (c>='a' && c<='f')
			return int(c-'a'+10);
		if (c>='A' && c<='F')
			return int(c-'A'+10);
		return -1;
	}

	std::string decode(const std::string &str){
		std::string res;
		int a,b;
		for (size_t i=0;i<str.size();++i){
			if (str[i]!='\\'){
				res+=str[i];
				continue;
			}
			++i;
			if (i==str.size())
				return res;
			switch(str[i]){
				case '\\':
					res+='\\';
					continue;
				case 'n':
					res+='\n';
					continue;
				case 'r':
					res+='\r';
					continue;
				case 't':
					res+='\t';
					continue;
				case '0':
					res+='\0';
					continue;
				case 'X':
				case 'x':
					if (i+2>=str.size())
						return res;
					a=hex2num(str[i+1]);
					b=hex2num(str[i+2]);
					if (a<0||b<0)
						return res;
					res+=char(a*16+b);
					i+=2;
					continue;
			}
		}
		return res;
	}

	std::string encode_hex(const std::string &str){
		std::string res(str.size()*2,'\0');
		for (size_t i=0;i<str.size();++i){
			res[2*i+0]="0123456789abcdef"[(((int)(unsigned char)str[i])>>4)&0xf];
			res[2*i+1]="0123456789abcdef"[(((int)(unsigned char)str[i])>>0)&0xf];
		}
		return res;
	}

	std::string encode_HEX(const std::string &str){
		std::string res(str.size()*2,'\0');
		for (size_t i=0;i<str.size();++i){
			res[2*i+0]="0123456789ABCDEF"[(((int)(unsigned char)str[i])>>4)&0xf];
			res[2*i+1]="0123456789ABCDEF"[(((int)(unsigned char)str[i])>>0)&0xf];
		}
		return res;
	}

	std::string decode_hex(const std::string &str){
		assertErr(str.size()%2==0)<<OUTPUT(str.size(),alt_print(str));
		std::string res(str.size()/2,'\0');
		for (size_t i=0;i<res.size();++i){
			int a=hex2num(str[2*i+0]);
			int b=hex2num(str[2*i+1]);
			assertErr(a>=0 && b>=0)<<OUTPUT(i,a,b,alt_print(str));
			res[i]=char(a*16+b);
		}
		return res;
	}
	template<>
	void _alt_print<std::string>::print_me(std::ostream &out)const{
		std::ostringstream res;
		//char sep=(std::count(t.begin(),t.end(),'\'')>std::count(t.begin(),t.end(),'"'))?'"':'\'';
		char sep='"';
		res.put(sep);
		encode_into(res,t,sep);
		res.put(sep);
		out<<res.str();
	}

	template<>
	void _alt_print<const char *>::print_me(std::ostream &out)const{
		if (!t){
			out<<"NULL";
			return;
		}
		std::ostringstream res;
		//char sep=(std::count(t.begin(),t.end(),'\'')>std::count(t.begin(),t.end(),'"'))?'"':'\'';
		char sep='"';
		res.put(sep);
		for(const char *c=t;*c;++c)
			add_encoded_char(res,*c,sep);
		res.put(sep);
		out<<res.str();
	}

	template<>
	void _alt_print<char *>::print_me(std::ostream &out)const{
		_alt_print<const char *>(this->t).print_me(out);
		return;
	}

	//==============================
	size_t _func_timer::depth=0;

	std::string sec2text(uint64_t t){
		std::ostringstream out;
		if (t>60*60*24)
			out<<t/(60*60*24)<<" days, ";
		t%=60*60*24;
		out<<STR("02",t/(60*60))<<":";
		t%=60*60;
		out<<STR("02",t/60)<<":";
		t%=60;
		out<<STR("02",t);
		return out.str();
	}

	std::string sec2textf(double t){
		std::ostringstream out;
		out<<sec2text(uint64_t(t));
		out<<"."<<STR("01",uint64_t(t*10)%10);
		return out.str();
	}

	std::string as_logn(double x,unsigned n){
		if (n<2)
			return STR(x);
		std::ostringstream out;
		double a=log((double)x)/log(n);
		int64_t b=a;
		if (b>a) --b;
		a=pow((double)n,b);
		out<<STR(".3f",x/a)<<"*"<<n<<"^"<<b;
		return out.str();
	}
	std::string as_logn(uint64_t x,unsigned n){
		if (n<2)
			return STR(x);
		std::ostringstream out;
		double a=log((double)x)/log(n);
		int64_t b=a;
		if (b>a) --b;
		a=pow((double)n,b);
		out<<STR(".3f",x/a)<<"*"<<n<<"^"<<b;
		return out.str();
	}

	time_counter::time_counter(uint64_t _total,const std::string &_name):
		count(0),
		total(_total),
		estimated(1),
		diff(1),
		last_time(0),
		name(_name),
		initialized(false)
	{}
	void time_counter::update_timer(){
		double t=timer()-T0;
		if (t>last_time+0.5){
			if (!initialized){
				if (updater_p)id=updater_p->init(total,name);
				initialized=true;
			}
			if (updater_p)updater_p->update(id,count,total,t);
			last_time=t;
			diff=(diff+1)/2;
			estimated=count+diff;
		}else{
			estimated=count+diff;
			diff=diff*2;
		}
	}
	void time_counter::start(){
			if (!initialized){
				if (count==0)
					T0.now();
				if (updater_p)id=updater_p->init(total,name);
				initialized=true;
			}
	}
	void time_counter::done(){
		if (initialized){
			if (updater_p)updater_p->done(id,count,timer()-T0);
		}
		initialized=false;
	}

	time_counter::updater *time_counter::updater_p=default_updater();
	time_counter::updater *time_counter::default_updater(){return console_updater();}

	unsigned time_counter::updater::logn=0;

	std::string time_counter::updater::text_init(uint64_t total){
		std::ostringstream out;
		out<<"Starting counter";
		if (total)
			out<<" "<<total<<" iterations";
		return out.str();
	}
	std::string time_counter::updater::text_update(uint64_t count, uint64_t total, double t){
		std::ostringstream out;
		if (total)
			out<<as_logn(count,logn)<<"/"<<as_logn(total,logn);
		else
			out<<as_logn(count,logn);
		out<<" done";
		if (total)
			out<<" ("<<STR(".1f",100.*count/total)<<"%)";
		out<<". Elapsed "<<sec2textf(t);
		{
			double a;
			const char *b;
			if (t<count){
				a=count/t;
				b="second";
			}else if (t<60*count){
				a=60*count/t;
				b="minute";
			}else{
				a=3600*count/t;
				b="hour";
			}
				out<<" ("<<as_logn(a,logn)<<" /"<<b<<").";
		}
		if (total)
			out<<" Left "<<sec2textf((total-count)*t/count)<<".";
		return out.str();
	}
	std::string time_counter::updater::text_done(uint64_t count, double t){
		std::ostringstream out;
		out<<"Done! ";
		out<<as_logn(count,logn);
		out<<" iterations took "<<sec2textf(t);
		{
			double a;
			const char *b;
			if (t<count){
				a=count/t;
				b="second";
			}else if (t<60*count){
				a=60*count/t;
				b="minute";
			}else{
				a=3600*count/t;
				b="hour";
			}
			out<<" ("<<as_logn(a,logn)<<" /"<<b<<").";
		}
		return out.str();
	}

	namespace{
		class _console_updater:public time_counter::updater{
			public:
				_console_updater(bool l2=false):log2(l2),exists(false),len(0){}
				virtual size_t init(uint64_t total, const std::string &name);
				virtual void update(size_t id, uint64_t count, uint64_t total, double t);
				virtual void done(size_t id, uint64_t count, double t);
				bool log2;
			private:
				std::string name;
				bool exists;
				size_t len;
		};
	}
	time_counter::updater *console_updater(){
		static _console_updater res(false);
		return &res;
	}
	time_counter::updater *console_updater_log(){
		static _console_updater res(true);
		return &res;
	}
	namespace{
		size_t _console_updater::init(uint64_t total, const std::string &_name){
			if (exists) return 1;
			exists=true;
			name=_name;
			std::string str=name;
			if (name.size()) str+=": ";
			str+=text_init(total);
			len=str.size();
			OUT << str<<std::flush;
			return 0;
		}
		void _console_updater::update(size_t id, uint64_t count, uint64_t total, double t){
			if (id) return;
			OUT<<'\r';
			for (size_t i=0;i<len;++i)OUT<<' ';
			OUT<<'\r';
			std::string str=name;
			if (name.size()) str+=": ";
			str+=text_update(count,total,t);
			len=str.size();
			OUT<<str<<std::flush;
		}
		void _console_updater::done(size_t id,uint64_t count, double t){
			if (id) return;
			exists=false;
			OUT<<'\r';
			for (size_t i=0;i<len;++i)OUT<<' ';
			OUT<<'\r';
			std::string str=name;
			if (name.size()) str+=": ";
			str+=text_done(count,t);
			OUT<<str<<std::endl;
		}
	}

	std::vector<std::string> glob(const char *pattern){
		Glob g(pattern);
		return std::vector<std::string>(g.begin(),g.end());
	}

	struct Glob::internal_t{
		glob_t g;
	};

	Glob::Glob(const char *pattern){
		data=new internal_t;
		data->g.gl_offs=0;
		if (::glob(pattern,GLOB_MARK | GLOB_NOESCAPE | GLOB_BRACE | GLOB_TILDE_CHECK,NULL,&data->g)!=0)
			data->g.gl_pathc=0;
	}
	Glob::~Glob(){
		::globfree(&data->g);
		delete data;
	}
	size_t Glob::size()const{return data->g.gl_pathc;}
	const char *const*Glob::begin()const{return data->g.gl_pathv;}
	const char *const*Glob::end()const{return data->g.gl_pathv+data->g.gl_pathc;}
}

