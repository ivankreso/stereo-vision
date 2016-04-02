#include "config_parser.h"

#include <algorithm>
#include <sstream>
#include <iostream>

#include <boost/algorithm/string.hpp>

namespace track {

ConfigParserVector configParser(std::string conf){
  ConfigParserVector confVec;

  size_t posbeg=0 ;
  while (1){
    posbeg=conf.find_first_not_of(" ", posbeg);
    if (posbeg>=conf.size()){
      break;
    }
    int depth=0;
    auto i=posbeg;
    while (i<conf.size()){
      if (conf[i]=='[') ++depth; 
      if (conf[i]==']') --depth; 
      if (conf[i]==' ' && depth==0)
        break;
      ++i;
    }
    auto posend=i;
    auto poseq=conf.find("=", posbeg);

    std::string key;
    if (poseq<posend){
      key=std::string(conf, posbeg,  poseq-posbeg);
      posbeg=poseq+1;
    }
    std::string val;
    if (conf[posbeg]=='['){
      val=std::string(conf, posbeg+1, posend-posbeg-2);
    } else{
      val=std::string(conf, posbeg, posend-posbeg);
    }
    confVec.push_back(std::make_pair(key,val));

    posbeg=posend;
  }

  return confVec;      
}

ConfigParserMap configParserVecToMap(
  ConfigParserVector confVec, 
  ConfigParserKeys keys)
{
  ConfigParserMap confMap;

  for (size_t i=0; i<confVec.size(); ++i){
    auto& kv=confVec[i];
    if (!kv.first.empty()){
      confMap.insert(std::make_pair(kv.first, kv.second));
    } else{
      confMap.insert(std::make_pair(keys[i], kv.second));
    }
  }

  for (auto x : keys){
    std::string strx(x);
    auto it = confMap.lower_bound(strx);
    if (it != confMap.end() && strx!=it->first) {
      confMap.insert(it, std::make_pair(strx, std::string("")));
    }
  }

  return confMap;
}

ConfigParserMap configParser(
  std::string conf, ConfigParserKeys keys)
{
  return configParserVecToMap(configParser(conf), keys);
}

void configParserMerge(
  ConfigParserMap confSrc, 
  ConfigParserKeys keys,
  ConfigParserMap& confDst) 
{
  for (auto x: keys){
    std::string strx(x);
    auto it = confSrc.find(strx);
    if (it != confSrc.end() && !it->second.empty()) {
      confDst[strx]=it->second;
    }
  }
}

std::string configParserJoin(ConfigParserKeys keys){
  std::string rv;
  for (auto s : keys){
    rv+= s;
    rv+= ' ';
  }
  if (rv.size()>0){
    rv.erase(rv.end()-1);
  }
  return rv;
}

std::string configParserReport(
  ConfigParserMap confMap, 
  ConfigParserKeys keys) 
{
  std::ostringstream oss;
  for (auto x : keys){
    std::string strx(x);
    oss <<strx <<"=" <<confMap[strx] <<" ";
  }
  int szfix=std::max(0, int(oss.str().size())-1);
  return std::string(oss.str(), 0, szfix);
}

namespace {
  int test(std::string s) {
    std::cout <<"Testing: " <<s <<"\n";

    auto confVec=configParser(s);
    std::cout <<"confVec: \n";
    for (auto& kv: confVec){
      std::cout <<"  " <<kv.first <<": '" <<kv.second <<"'\n";
    }

    char const* keysArray[]={"a", "b", "c", "d"};
    ConfigParserKeys keys(&keysArray[0], 
        &keysArray[0]+sizeof(keysArray)/sizeof(keysArray[0]) );
    auto confMap=configParserVecToMap(confVec, keys);
    for (auto& kv: confMap){
      std::cout <<"  " <<kv.first <<": '" <<kv.second <<"'\n";
    }

    return 0;
  }

  //int bla=test("a=5 b=[3 4 5] c=[1 [2 [3 4]]]");
  //int bla2=test("a=5 b=[3 4 5] [1 [2 [3 4]]] x=");
}

}
