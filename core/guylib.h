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

#ifndef GUYLIB_H__
#define GUYLIB_H__

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <list>
#include <exception>
#include <cstdlib>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>

#if defined(__GXX_EXPERIMENTAL_CXX0X) || __cplusplus >= 201103L
#define GUYLIB_CPP_11
#endif

namespace guylib{

	struct assertException:public std::exception{
		assertException(const std::string &str)throw():msg(str){}
		~assertException()throw(){}
		const char *what()const throw(){return msg.c_str();}
		std::string msg;
	};

	struct assertClass{
		assertClass(bool iexit):_exit(iexit),first(true){}
		void apply();
		std::ostream &out(){return _out;}
		bool _exit;
		bool first;
		std::ostringstream _out;
		static bool _throw;
		static std::ostream *deb_out;
	};

	inline void assertErr_throw(bool b=true){assertClass::_throw=b;}
	inline std::ostream *debug_output(std::ostream *out){std::swap(out,assertClass::deb_out);return out;}
	struct debug_output_guard{
		debug_output_guard():aquired(false){}
		debug_output_guard(std::ostream *_out):aquired(false){
			aquire(_out);
		}
		bool aquire(std::ostream *_out){
			if (aquired)
				return false;
			prev=debug_output(_out);
			aquired=true;
			return true;
		}
		bool release(){
			if (!aquired)
				return false;
			debug_output(prev);
			aquired=false;
			return true;
		}
		~debug_output_guard(){release();}
		private:
			std::ostream *prev;
			bool aquired;
	};

#define LINE_FILE __FILE__<<':'<<__LINE__<<": in "<<__PRETTY_FUNCTION__<<": "
#define assertErr(x) while(!(x))for(guylib::assertClass assertErrXXX(true);assertErrXXX.first;assertErrXXX.apply()) assertErrXXX.out()<<LINE_FILE<<" assertErr("<<#x<<") Failed!\n"
#define DEB for(guylib::assertClass assertErrXXX(false);assertErrXXX.first;assertErrXXX.apply()) assertErrXXX.out()<<LINE_FILE<<'\n'
#define OUT if (!guylib::assertClass::deb_out){}else (*guylib::assertClass::deb_out)

#define LOCATION_STR (std::string(__FILE__)+":"+guylib::STR(__LINE__)+": in "+guylib::STR(__PRETTY_FUNCTION__))

#ifdef GUYLIB_CPP_11
	template <class A, class R>
	class _guard{
		public:
			_guard(A a, R r):_r(r),_aquired(true){a();}
			~_guard(){release();}
			_guard(_guard &&o):_r(std::move(o._r)),_aquired(o._aquired){o._aquired=false;}
			void release(){if (_aquired) _r();_aquired=false;}
			bool aquired()const{return _aquired;}
		private:
			_guard(const _guard &)=delete;
			void operator=(const _guard &)=delete;
			void operator=(_guard &&o)=delete;
			R _r;
			bool _aquired;
	};

	template <class A, class R>
	_guard<A,R> guard(A a, R r){return _guard<A,R>(a,r);}
#endif

	struct sstate{
		sstate();
		sstate(const std::string &pat);
		sstate(const char *pat);
		sstate(const std::ostream &s);
		void clear();
		bool from(const char *pat);
		bool from(const std::string &pat){return from(pat.c_str());}
		void from(const std::ios &s);
		void to(std::ios &s)const;
		std::string str()const;
		struct save{
			save(sstate &_s):s(_s){}
			sstate &s;
		};
		std::streamsize precision;
		std::streamsize width;
		std::ios::char_type fill;
		std::ios::fmtflags flags;
	};
	inline std::ostream &operator<<(std::ostream &out,const sstate &s){
		s.to(out);
		return out;
	}
	inline std::ostream &operator<<(std::ostream &out,sstate::save &s){
		s.s.from(out);
		return out;
	}

	class sstate_guard{
		public:
			sstate_guard(std::ostream &o):out(o),s(o){}
			~sstate_guard(){s.to(out);}
			std::ostream &out;
		private:
			sstate s;
	};

	template<class T>
	std::string STR(const T&t){
		std::ostringstream out;
		out<<t;
		return out.str();
	}

	template<class T>
	std::string STR(const sstate &s, const T&t){
		std::ostringstream out;
		out<<s<<t;
		return out.str();
	}

#define _OUTPUT_CNT(...) _OUTPUT_CNT2(__VA_ARGS__,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define _OUTPUT_CNT2(a20,a19,a18,a17,a16,a15,a14,a13,a12,a11,a10,a9,a8,a7,a6,a5,a4,a3,a2,a1,a0,...) a0
#define _OUTPUT_RUN(func,...) _OUTPUT_RUN2(func,_OUTPUT_CNT(__VA_ARGS__))
#define _OUTPUT_RUN2(func,n) _OUTPUT_RUN3(func,n)
#define _OUTPUT_RUN3(func,n) func ## n

#define OUTPUT_M(x) #x<<"= "<<guylib::STR(guylib::alt_print(x))<<"; "
#define OUTPUT_1(x) #x<<"= "<<guylib::STR(guylib::alt_print(x))<<';'<<std::endl
#define OUTPUT_2(x,...) OUTPUT_M(x)<<OUTPUT_1(__VA_ARGS__)
#define OUTPUT_3(x,...) OUTPUT_M(x)<<OUTPUT_2(__VA_ARGS__)
#define OUTPUT_4(x,...) OUTPUT_M(x)<<OUTPUT_3(__VA_ARGS__)
#define OUTPUT_5(x,...) OUTPUT_M(x)<<OUTPUT_4(__VA_ARGS__)
#define OUTPUT_6(x,...) OUTPUT_M(x)<<OUTPUT_5(__VA_ARGS__)
#define OUTPUT_7(x,...) OUTPUT_M(x)<<OUTPUT_6(__VA_ARGS__)
#define OUTPUT_8(x,...) OUTPUT_M(x)<<OUTPUT_7(__VA_ARGS__)
#define OUTPUT_9(x,...) OUTPUT_M(x)<<OUTPUT_8(__VA_ARGS__)
#define OUTPUT_10(x,...) OUTPUT_M(x)<<OUTPUT_9(__VA_ARGS__)
#define OUTPUT_11(x,...) OUTPUT_M(x)<<OUTPUT_10(__VA_ARGS__)
#define OUTPUT_12(x,...) OUTPUT_M(x)<<OUTPUT_11(__VA_ARGS__)
#define OUTPUT_13(x,...) OUTPUT_M(x)<<OUTPUT_12(__VA_ARGS__)
#define OUTPUT_14(x,...) OUTPUT_M(x)<<OUTPUT_13(__VA_ARGS__)
#define OUTPUT_15(x,...) OUTPUT_M(x)<<OUTPUT_14(__VA_ARGS__)
#define OUTPUT_16(x,...) OUTPUT_M(x)<<OUTPUT_15(__VA_ARGS__)
#define OUTPUT_17(x,...) OUTPUT_M(x)<<OUTPUT_16(__VA_ARGS__)
#define OUTPUT_18(x,...) OUTPUT_M(x)<<OUTPUT_17(__VA_ARGS__)
#define OUTPUT_19(x,...) OUTPUT_M(x)<<OUTPUT_18(__VA_ARGS__)
#define OUTPUT(...) _OUTPUT_RUN(OUTPUT_,__VA_ARGS__)(__VA_ARGS__)

#define OUTPUTF_M(s,x) #x<<"= "<<guylib::STR(s,guylib::alt_print(x))<<"; "
#define OUTPUTF_1(s,x) #x<<"= "<<guylib::STR(s,guylib::alt_print(x))<<';'<<std::endl
#define OUTPUTF_2(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_1(s,__VA_ARGS__)
#define OUTPUTF_3(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_2(s,__VA_ARGS__)
#define OUTPUTF_4(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_3(s,__VA_ARGS__)
#define OUTPUTF_5(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_4(s,__VA_ARGS__)
#define OUTPUTF_6(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_5(s,__VA_ARGS__)
#define OUTPUTF_7(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_6(s,__VA_ARGS__)
#define OUTPUTF_8(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_7(s,__VA_ARGS__)
#define OUTPUTF_9(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_8(s,__VA_ARGS__)
#define OUTPUTF_10(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_9(s,__VA_ARGS__)
#define OUTPUTF_11(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_10(s,__VA_ARGS__)
#define OUTPUTF_12(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_11(s,__VA_ARGS__)
#define OUTPUTF_13(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_12(s,__VA_ARGS__)
#define OUTPUTF_14(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_13(s,__VA_ARGS__)
#define OUTPUTF_15(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_14(s,__VA_ARGS__)
#define OUTPUTF_16(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_15(s,__VA_ARGS__)
#define OUTPUTF_17(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_16(s,__VA_ARGS__)
#define OUTPUTF_18(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_17(s,__VA_ARGS__)
#define OUTPUTF_19(s,x,...) OUTPUTF_M(s,x)<<OUTPUTF_18(s,__VA_ARGS__)
#define OUTPUTF(s,...) _OUTPUT_RUN(OUTPUTF_,__VA_ARGS__)(s,__VA_ARGS__)

#define OUTPUTFF_M(s,x) #x<<"= "<<guylib::STR(s,guylib::alt_print(x))<<"; "
#define OUTPUTFF_2(s,x) #x<<"= "<<guylib::STR(s,guylib::alt_print(x))<<';'<<std::endl
#define OUTPUTFF_4(s,x,...) OUTPUTF_M(s,x)<<OUTPUTFF_2(__VA_ARGS__)
#define OUTPUTFF_6(s,x,...) OUTPUTF_M(s,x)<<OUTPUTFF_4(__VA_ARGS__)
#define OUTPUTFF_8(s,x,...) OUTPUTF_M(s,x)<<OUTPUTFF_6(__VA_ARGS__)
#define OUTPUTFF_10(s,x,...) OUTPUTF_M(s,x)<<OUTPUTFF_8(__VA_ARGS__)
#define OUTPUTFF_12(s,x,...) OUTPUTF_M(s,x)<<OUTPUTFF_10(__VA_ARGS__)
#define OUTPUTFF_14(s,x,...) OUTPUTF_M(s,x)<<OUTPUTFF_12(__VA_ARGS__)
#define OUTPUTFF_16(s,x,...) OUTPUTF_M(s,x)<<OUTPUTFF_14(__VA_ARGS__)
#define OUTPUTFF_18(s,x,...) OUTPUTF_M(s,x)<<OUTPUTFF_16(__VA_ARGS__)
#define OUTPUTFF(...) _OUTPUT_RUN(OUTPUTFF_,__VA_ARGS__)(__VA_ARGS__)

	template<class T>
	struct _alt_print{
		_alt_print(const T&_t):t(_t){}
		void print_me(std::ostream &out)const{out<<t;}
		const T&t;
	};

	template<class T>
	_alt_print<T> alt_print(const T&t){return _alt_print<T>(t);}

	template<class T>
	std::ostream &operator<<(std::ostream &out,const _alt_print<T> &a){
		a.print_me(out);
		return out;
	}

	template<class T1, class T2>
	std::ostream &operator<<(std::ostream &out,const std::pair<T1,T2> &p){
		sstate s(out),c;
		return out<<c<<'('<<s<<alt_print(p.first)<<c<<','<<s<<alt_print(p.second)<<c<<')'<<s;
	}

	template<class ITER>
	struct _container_out{
		_container_out(ITER _b, ITER _e, size_t _max, bool _down):b(_b),e(_e),max(_max),down(_down){}
		void print_down(std::ostream &out)const{
			sstate s(out);
			sstate c;
			if (b==e){
				out <<c<< "[empty vector]"<<s;
				return;
			}
			size_t width=0;
			size_t size=std::distance(b,e);
			for(size_t num=size-1;num;num/=10)
				width+=1;
			if (!width) width=1;
			size_t cnt=0;
			for (ITER i=b;i!=e;++i,++cnt){
				if (cnt>=max && cnt!=size-1) continue;
				out<<c<<'\n';
				if (cnt==size-1 && max<cnt) out<<"...\n";
				out.width(width);
				out<<cnt<<": "<<s<<alt_print(*i);
			}
		}
		void print_vec(std::ostream &out)const{
			sstate s(out);
			sstate c;
			if (b==e){
				out << c<<"[]"<<s;
				return;
			}
			size_t size=std::distance(b,e);
			size_t cnt=0;
			out<<c<<'[';
			for (ITER i=b;i!=e;++i,++cnt){
				if (cnt>=max && cnt!=size-1) continue;
				out<<c;
				if (i!=b)
					out<<", ";
				if (cnt==size-1 && max<cnt) out<<"..., ";
				out<<s<<alt_print(*i);
			}
			out<<c<<']'<<s;
		}
		void print_me(std::ostream &out)const{
			if (down)
				print_down(out);
			else
				print_vec(out);
		}
		ITER b,e;
		size_t max;
		bool down;
	};
	template<class ITER>
	std::ostream &operator<<(std::ostream &out, const _container_out<ITER> &c){
		c.print_me(out);
		return out;
	}
	template<class ITER>
	_container_out<ITER> vec_out(ITER b, ITER e, size_t n=size_t(-1)){return _container_out<ITER>(b,e,n,false);}
	template<class ITER>
	_container_out<ITER> vec_down(ITER b, ITER e, size_t n=size_t(-1)){return _container_out<ITER>(b,e,n,true);}
	template<class C>
	_container_out<typename C::const_iterator> vec_out(const C&c, size_t n=size_t(-1)){return _container_out<typename C::const_iterator>(c.begin(),c.end(),n,false);}
	template<class C>
	_container_out<typename C::const_iterator> vec_down(const C&c, size_t n=size_t(-1)){return _container_out<typename C::const_iterator>(c.begin(),c.end(),n,true);}

	template<class T, class A>
	std::ostream &operator<<(std::ostream &out, const std::vector<T,A> &v){return out<<vec_out(v);}

	template<class K, class T, class C, class A>
	std::ostream &operator<<(std::ostream &out, const std::map<K,T,C,A> &v){
		typedef typename std::map<K,T,C,A>::const_iterator ITER;
		ITER b=v.begin();
		ITER e=v.end();
		sstate s(out);
		sstate c;
		if (b==e){
			out << c<<"{}"<<s;
			return out;
		}
		size_t cnt=0;
		out<<c<<'{';
		for (ITER i=b;i!=e;++i,++cnt){
			out<<c;
			if (i!=b)
				out<<", ";
			out<<s<<alt_print(i->first)<<c<<": "<<s<<alt_print(i->second);
		}
		out<<c<<'}'<<s;
		return out;
	}

	void add_encoded_char(std::ostream &out, char c, char sep);

	void encode_into(std::ostream &out, const std::string &str, char sep);

	std::string encode(const std::string &str, char sep);

	int hex2num(char c);

	std::string decode(const std::string &str);

	std::string encode_hex(const std::string &str);
	std::string encode_HEX(const std::string &str);
	std::string decode_hex(const std::string &str);

	template<>
	void _alt_print<std::string>::print_me(std::ostream &out)const;

	template<>
	void _alt_print<char *>::print_me(std::ostream &out)const;

	template<>
	void _alt_print<const char *>::print_me(std::ostream &out)const;

	//==============================

	template<class T>
	std::string type_name(){
		std::string s= __PRETTY_FUNCTION__;
		size_t a=s.find('=');
		assertErr(a!=std::string::npos && a<s.size()-2 && s[a+1] == ' ')<<OUTPUT(s)<<OUTPUT(a);
		a+=2;
		size_t b=s.find(';',a);
		assertErr(b!=std::string::npos)<<OUTPUT(s)<<OUTPUT(a)<<OUTPUT(b);
		return s.substr(a,b-a);
	}

	template<class T>
	std::string type_name(const T&t){
		return type_name<T>();
	}

	class timer{
		public:
			timer(){now();}
			void now(){
				gettimeofday(&tv,NULL);
			}
			double reset(){
				timer t;
				double res=t-(*this);
				*this=t;
				return res;
			}
			double operator-(const timer &o)const{
				return tv.tv_sec-o.tv.tv_sec+(tv.tv_usec-o.tv.tv_usec)*1.e-6;
			}
		private:
			timeval tv;
	};

	class _func_timer{
		public:
			_func_timer(const char *file, int line, const char *function){
				for (size_t i=0;i<depth;++i)
					OUT<<' ';
				out<<"FUNCTION TIMER - "<<file<<':'<<line<<": "<<function;
				OUT<<"> "<<out.str()<<std::endl;
				++depth;
				T.now();
			}
			~_func_timer(){
				double t=T.reset();
				--depth;
				for (size_t i=0;i<depth;++i)
					OUT<<' ';
				OUT<<"< "<<out.str()<<" Took "<<t<<" seconds"<<std::endl;
			}
		private:
			static size_t depth;
			timer T;
			std::ostringstream out;
	};

#define FUNC_TIMER _func_timer XXX_FUNC_TIMER_XXX(__FILE__,__LINE__,__PRETTY_FUNCTION__)

	std::string sec2text(uint64_t t);

	std::string sec2textf(double t);

	std::string as_logn(double x,unsigned n);
	std::string as_logn(uint64_t x,unsigned n);

	class time_counter{
		public:
			time_counter(uint64_t _total=0,const std::string &name="");
			void operator++(){
				++count;
				if (count<estimated) return;
				update_timer();
			}
			void start();
			void done();
			uint64_t get_total()const{return total;}
			void set_total(uint64_t t){total=t;}
		public:
			class updater{
				public:
					virtual size_t init(uint64_t total,const std::string &name)=0;
					virtual void update(size_t id, uint64_t count, uint64_t total, double t)=0;
					virtual void done(size_t id, uint64_t count, double t)=0;
					static std::string text_init(uint64_t total);
					static std::string text_update(uint64_t count, uint64_t total, double t);
					static std::string text_done(uint64_t count, double t);
					static void set_log(unsigned n){logn=n;}
					static unsigned get_log(){return logn;}
				private:
					static unsigned logn;
			};
			static void set_updater(updater *u){updater_p=u;}
			static updater *default_updater();
		private:
			void update_timer();
			uint64_t count,total;
			uint64_t estimated,diff;
			double last_time;
			timer T0;
			std::string name;
			size_t id;
			bool initialized;

		private:
			static updater *updater_p;
	};
	time_counter::updater *console_updater();
	time_counter::updater *console_updater_log();

	template<class ITER>
	struct progress_iter_t:public std::iterator<
										typename std::forward_iterator_tag,
										typename std::iterator_traits<ITER>::value_type,
										typename std::iterator_traits<ITER>::difference_type,
										typename std::iterator_traits<ITER>::pointer,
										typename std::iterator_traits<ITER>::reference>
	{
		public:
			progress_iter_t(ITER _iter,uint64_t _total=0, const std::string &name=""):iter(_iter),tc(_total,name),do_done(false){tc.start();}
			// used for the "end", doesn't start the counter
			progress_iter_t(ITER _iter,std::input_iterator_tag):iter(_iter),do_done(false){}
			~progress_iter_t(){if(do_done)tc.done();}
			progress_iter_t &operator++(){
				++tc;
				++iter;
				do_done=true;
				return *this;
			}
			progress_iter_t operator++(int){
				progress_iter_t res=*this;
				++(*this);
				return res;
			}
			inline bool operator==(const progress_iter_t &o){return iter==o.iter;}
			inline bool operator!=(const progress_iter_t &o){return iter!=o.iter;}
			friend inline bool operator==(progress_iter_t &a, ITER b){a.setup(b,std::__iterator_category(b));return a.iter==b;}
			friend inline bool operator!=(progress_iter_t &a, ITER b){a.setup(b,std::__iterator_category(b));return a.iter!=b;}
			friend inline bool operator==(ITER b, progress_iter_t &a){a.setup(b,std::__iterator_category(b));return a.iter==b;}
			friend inline bool operator!=(ITER b, progress_iter_t &a){a.setup(b,std::__iterator_category(b));return a.iter!=b;}
			typename std::iterator_traits<ITER>::reference operator*(){return *iter;}
			ITER operator->(){return iter;}

		private:
			inline void setup(const ITER &b,std::input_iterator_tag){
			}
			inline void setup(const ITER &b,std::random_access_iterator_tag){
				if (tc.get_total()) return;
				tc.set_total(b-iter);
			}
			ITER iter;
			time_counter tc;
			bool do_done;
	};

	template<class ITER>
	progress_iter_t<ITER> progress_iter(ITER i, const std::string &name="", uint64_t _total=0){return progress_iter_t<ITER>(i,_total,name);}
	template<class ITER>
	progress_iter_t<ITER> progress_end(ITER i){return progress_iter_t<ITER>(i,std::input_iterator_tag());}

	template<class ITER>
	class _range{
		public:
			typedef ITER iterator;
			typedef ITER const_iterator;
			_range(ITER _b, ITER _e):b(_b),e(_e),s(0){}
			ITER begin()const{return b;}
			ITER end()const{return e;}
			typename std::iterator_traits<ITER>::difference_type size()const{if (s==0) return s=std::distance(b,e); return s;}
			template<class C>
			operator C()const{return C(begin(),end());}
		private:
			ITER b,e;
			mutable typename std::iterator_traits<ITER>::difference_type s;
	};

	template<class ITER>
	inline _range<ITER> make_range(ITER b, ITER e){return _range<ITER>(b,e);}
	template<class ITER>
	inline _range<ITER> make_range(std::pair<ITER,ITER> p){return _range<ITER>(p.first,p.second);}

	template<class ITER>
	_range<progress_iter_t<ITER> > progress(ITER b, ITER e,uint64_t size,const std::string &name=""){return make_range(progress_iter(b,name,size),progress_end(e));}
	template<class ITER>
	_range<progress_iter_t<ITER> > progress(ITER b, ITER e,const std::string &name,std::input_iterator_tag){return progress(b,e,0,name);}
	template<class ITER>
	_range<progress_iter_t<ITER> > progress(ITER b, ITER e,const std::string &name,std::random_access_iterator_tag){return progress(b,e,e-b,name);}
	template<class ITER>
	_range<progress_iter_t<ITER> > progress(ITER b, ITER e,const std::string &name=""){return progress(b,e,name,std::__iterator_category(b));}

	template<class C>
	_range<progress_iter_t<typename C::const_iterator> > progress(const C&c,const std::string &name=""){
		return progress(c.begin(),c.end(),c.size(),name);
	}
	template<class C>
	_range<progress_iter_t<typename C::iterator> > progress(C&c,const std::string &name=""){
		return progress(c.begin(),c.end(),c.size(),name);
	}

	template<class NUM>
	class progress_num{
		public:
			progress_num(NUM _n,const std::string &name=""):n(_n),tc(0,name){tc.start();}
			~progress_num(){tc.done();}
			NUM operator++(){++n;++tc;return n;}
			NUM operator++(int){NUM r=n;++n;++tc;return r;}
			inline operator NUM()const{return n;}
			inline NUM num()const{return n;}

			template<class N> inline friend bool operator==(progress_num &a, const N &b){a.setup(b);return a.n==b;}
			template<class N> inline friend bool operator!=(progress_num &a, const N &b){a.setup(b);return a.n!=b;}
			template<class N> inline friend bool operator<=(progress_num &a, const N &b){a.setup(b);return a.n<=b;}
			template<class N> inline friend bool operator>=(progress_num &a, const N &b){a.setup(b);return a.n>=b;}
			template<class N> inline friend bool operator< (progress_num &a, const N &b){a.setup(b);return a.n< b;}
			template<class N> inline friend bool operator> (progress_num &a, const N &b){a.setup(b);return a.n> b;}
			template<class N> inline friend bool operator==(const N &b, progress_num &a){a.setup(b);return a.n==b;}
			template<class N> inline friend bool operator!=(const N &b, progress_num &a){a.setup(b);return a.n!=b;}
			template<class N> inline friend bool operator<=(const N &b, progress_num &a){a.setup(b);return a.n<=b;}
			template<class N> inline friend bool operator>=(const N &b, progress_num &a){a.setup(b);return a.n>=b;}
			template<class N> inline friend bool operator< (const N &b, progress_num &a){a.setup(b);return a.n< b;}
			template<class N> inline friend bool operator> (const N &b, progress_num &a){a.setup(b);return a.n> b;}
		private:
			inline void setup(const NUM &b){
				if (tc.get_total()) return;
				if (n<b)
					tc.set_total(b-n);
				else
					tc.set_total(n-b);
			}
			NUM n;
			time_counter tc;
	};

	inline void progress_updater(time_counter::updater *u){time_counter::set_updater(u);}
	inline void progress_logn(unsigned n){time_counter::updater::set_log(n);}

	template<class NUM>
	struct increase_f{inline void operator()(NUM &n){++n;}};

	template<class NUM, class FUNC=increase_f<NUM> >
	struct _as_iter:public std::iterator<
										typename std::forward_iterator_tag,
										NUM,
										int64_t,
										const NUM*,
										const NUM&>
	{
		public:
			_as_iter(NUM n=NUM(),FUNC f=FUNC()):num(n),func(f){}
			_as_iter &operator++(){
				func(num);
				return *this;
			}
			_as_iter operator++(int){
				_as_iter res=*this;
				++(*this);
				return res;
			}
			inline bool operator==(const _as_iter &o)const{return num==o.num;}
			inline bool operator!=(const _as_iter &o)const{return num!=o.num;}
			const NUM &operator*()const{return num;}
			const NUM *operator->()const{return &num;}

		private:
			NUM num;
			FUNC func;
	};

	template<class NUM>
	_as_iter<NUM> make_iter(NUM n){return _as_iter<NUM>(n);}
	template<class NUM,class FUNC>
	_as_iter<NUM,FUNC> make_iter_f(NUM n,FUNC f){return _as_iter<NUM,FUNC>(n,f);}
	
	template<class NUM>
	_range<_as_iter<NUM> > range(NUM a, NUM b){return make_range(make_iter(a),make_iter((a<b)?b:a));}
	template<class NUM>
	_range<_as_iter<NUM> > range(NUM b){return range(NUM(0),b);}
	// template<class NUM,DIFF>
	// _range<_as_iter<NUM,increase_n_f<NUM,DIFF> > > range(NUM a, NUM b, DIFF d){b-=a;b/=d;b=(b>0)b:0;b*=d;b+=a;return make_range(make_iter(a,d),make_iter(b,d));}
	template<class NUM,class FUNC>
	_range<_as_iter<NUM,FUNC> > range_f(NUM a, NUM b, FUNC f){return make_range(make_iter(a,f),make_iter(b,f));}


	//--------------------------------

	template<class ITER>
	std::string join(ITER b, ITER e, const std::string &sep="",sstate s=sstate()){
		std::ostringstream res;
		sstate c;
		for (ITER i=b;i!=e;++i){
			if (i!=b)
				res<<c<<sep;
			res<<s<<*i;
		}
		return res.str();
	}

	template<class C>
	std::string join(const C&c, const std::string &sep="",sstate s=sstate()){
		return join(c.begin(),c.end(),sep,s);
	}

	template<class OITER>
	void split_iter(OITER res, const std::string &str,const std::string &sep=""){
		std::string::size_type loc=0;
		if (sep.empty())
			loc=str.find_first_not_of(" \t\n\r",loc);
		while(loc<str.size()){
			std::string::size_type next;
			if (sep.empty())
				next=str.find_first_of(" \t\n\r",loc);
			else
				next=str.find(sep,loc);
			if (next==std::string::npos)
				next=str.size();
			*res=str.substr(loc,next-loc);
			loc=next;
			if (sep.empty())
				loc=str.find_first_not_of(" \t\n\r",loc);
			else
				loc+=sep.size();
		}
	}

	inline std::vector<std::string> split(const std::string &str, char sep){
		std::vector<std::string> res;
		split_iter(std::back_inserter(res),str,std::string(1,sep));
		return res;
	}

	inline std::vector<std::string> split(const std::string &str, const std::string &sep=""){
		std::vector<std::string> res;
		split_iter(std::back_inserter(res),str,sep);
		return res;
	}

	//-------------------

	template<typename NUM,typename CHECK,typename ADVANCE>
	class _generator_iter:public std::iterator<
										typename std::forward_iterator_tag,
										NUM,
										int64_t,
										const NUM*,
										const NUM&>
	{
		public:
			_generator_iter(NUM n, CHECK c, ADVANCE a):_num(n),_chk(c),_adv(a),_done(!_chk(n)){}
			_generator_iter():_done(true){}
			bool operator==(const _generator_iter &o){return _done && o._done;}
			bool operator!=(const _generator_iter &o){return !(*this==o);}
			const NUM &operator*()const{return _num;}
			const NUM *operator->()const{return &_num;}
			_generator_iter &operator++(){_adv(_num);_done=!_chk(_num);return *this;}
			_generator_iter operator++(int){_generator_iter res=*this;++(*this);return res;}
			_generator_iter end()const{_generator_iter res=*this;res._done=true;return res;}
			bool done()const{return _done;}
		private:
			NUM _num;
			CHECK _chk;
			ADVANCE _adv;
			bool _done;
	};

	template<typename NUM,typename CHECK,typename ADVANCE>
	_range<_generator_iter<NUM,CHECK,ADVANCE> > generator(NUM n, CHECK c, ADVANCE a){
		_generator_iter<NUM,CHECK,ADVANCE> b(n,c,a);
		return make_range(b,b.end());
	}

	//-------------------

	class Glob{
		public:
			typedef const char *const* iterator;
			typedef const char *const* const_iterator;
		Glob(const char *pattern);
		~Glob();
		const_iterator begin()const;
		const_iterator end()const;
		size_t size()const;
		private:
		struct internal_t;
		internal_t *data;
	};

	inline std::ostream &operator<<(std::ostream &out,const Glob &g){
		return out<<guylib::vec_out(g.begin(),g.end());
	}

	std::vector<std::string> glob(const char *pattern);
	inline std::vector<std::string> glob(const std::string &pattern){return glob(pattern.c_str());}

	template<class T>
	inline T sqr(const T &t){return t*t;}
}

#endif

