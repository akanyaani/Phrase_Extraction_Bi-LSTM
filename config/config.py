from configparser import ConfigParser
import json
import os


class Config(object):
    """
    Config parser to load and return neural net configurations
    """
    def __init__(self, conf_file=None):
        """
        Initialize the class and set the config file property
        """
        self.config = ConfigParser()
        self.confFile = os.environ.get('CONFIG_FILE', "config/system.config")\
                        if conf_file else os.path.abspath(conf_file)
        if not os.path.isfile(self.confFile):
            raise Exception("%s : File does not exist..." % self.confFile)
        self.config.read(self.confFile)

    def getConfig(self, section, item):
        """
        Returns the property from the config file
        :returns: Config Object
        :rtype: Config Object
        """
        json_acceptable_string = self.config.get(section, item).replace("'", "\"")
        return json.loads(json_acceptable_string)


if __name__ == '__main__':
    pass

