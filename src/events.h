
#ifndef _EVENTS_H_
#define _EVENTS_H_

void mouseEventHandler(int button, int state, int x, int y);
void motionEventHandler(int x, int y);
void keyboardEventHandler(unsigned char key, int x, int y);
std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);
void control_cb(int control);

#endif