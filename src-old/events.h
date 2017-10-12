#ifndef _EVENTS_H_
#define _EVENTS_H_

void mouseEventHandler(int button, int state, int x, int y);
void motionEventHandler(int x, int y);
void keyboardEventHandler(unsigned char key, int x, int y);
void control_cb(int control);

#endif
