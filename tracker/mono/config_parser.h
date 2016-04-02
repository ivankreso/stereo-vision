#ifndef CONFIG_PARSER_H_
#define CONFIG_PARSER_H_

#include <string>
#include <map>
#include <vector>
#include <utility>

namespace track {

typedef std::vector<std::pair<std::string, std::string> > ConfigParserVector;
typedef std::map<std::string, std::string> ConfigParserMap;
typedef std::vector<char const*> ConfigParserKeys;

ConfigParserMap configParser(std::string confStr, ConfigParserKeys keys);

void configParserMerge(ConfigParserMap confSrc, ConfigParserKeys keys, ConfigParserMap& confDst);

std::string configParserJoin(ConfigParserKeys vec);

std::string configParserReport(ConfigParserMap confSrc, ConfigParserKeys keys);

ConfigParserVector configParser(std::string confStr);
ConfigParserMap configParserVecToMap(ConfigParserVector confVec, ConfigParserKeys keys);

}


#endif
