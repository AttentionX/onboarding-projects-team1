from pprint import pprint
from datasets import load_dataset

# https://huggingface.co/datasets/samsum
samsum_dataset = load_dataset('samsum')
pprint(samsum_dataset['train'])
pprint(samsum_dataset['test'])
pprint(samsum_dataset['validation'])

"""
Dataset({
    features: ['id', 'dialogue', 'summary'],
    num_rows: 14732
})
Dataset({
    features: ['id', 'dialogue', 'summary'],
    num_rows: 819
})
Dataset({
    features: ['id', 'dialogue', 'summary'],
    num_rows: 818
})
"""


# have a look at first 10 entries of the training set
for i in range(10):
    print(f"Dialogue: {samsum_dataset['train'][i]['dialogue']}")
    print(f"Summary: {samsum_dataset['train'][i]['summary']}")
    print("---")


"""
Dialogue: Amanda: I baked  cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)
Summary: Amanda baked cookies and will bring Jerry some tomorrow.
---
Dialogue: Olivia: Who are you voting for in this election? 
Oliver: Liberals as always.
Olivia: Me too!!
Oliver: Great
Summary: Olivia and Olivier are voting for liberals in this election. 
---
Dialogue: Tim: Hi, what's up?
Kim: Bad mood tbh, I was going to do lots of stuff but ended up procrastinating
Tim: What did you plan on doing?
Kim: Oh you know, uni stuff and unfucking my room
Kim: Maybe tomorrow I'll move my ass and do everything
Kim: We were going to defrost a fridge so instead of shopping I'll eat some defrosted veggies
Tim: For doing stuff I recommend Pomodoro technique where u use breaks for doing chores
Tim: It really helps
Kim: thanks, maybe I'll do that
Tim: I also like using post-its in kaban style
Summary: Kim may try the pomodoro technique recommended by Tim to get more stuff done.
---
Dialogue: Edward: Rachel, I think I'm in ove with Bella..
rachel: Dont say anything else..
Edward: What do you mean??
rachel: Open your fu**ing door.. I'm outside
Summary: Edward thinks he is in love with Bella. Rachel wants Edward to open his door. Rachel is outside. 
---
Dialogue: Sam: hey  overheard rick say something
Sam: i don't know what to do :-/
Naomi: what did he say??
Sam: he was talking on the phone with someone
Sam: i don't know who
Sam: and he was telling them that he wasn't very happy here
Naomi: damn!!!
Sam: he was saying he doesn't like being my roommate
Naomi: wow, how do you feel about it?
Sam: i thought i was a good rommate
Sam: and that we have a nice place
Naomi: that's true man!!!
Naomi: i used to love living with you before i moved in with me boyfriend
Naomi: i don't know why he's saying that
Sam: what should i do???
Naomi: honestly if it's bothering you that much you should talk to him
Naomi: see what's going on
Sam: i don't want to get in any kind of confrontation though
Sam: maybe i'll just let it go
Sam: and see how it goes in the future
Naomi: it's your choice sam
Naomi: if i were you i would just talk to him and clear the air
Summary: Sam is confused, because he overheard Rick complaining about him as a roommate. Naomi thinks Sam should talk to Rick. Sam is not sure what to do.
---
Dialogue: Neville: Hi there, does anyone remember what date I got married on?
Don: Are you serious?
Neville: Dead serious. We're on vacation, and Tina's mad at me about something. I have a strange suspicion that this might have something to do with our wedding anniversary, but I have nowhere to check.
Wyatt: Hang on, I'll ask my wife.
Don: Haha, someone's in a lot of trouble :D
Wyatt: September 17. I hope you remember the year ;)
Summary: Wyatt reminds Neville his wedding anniversary is on the 17th of September. Neville's wife is upset and it might be because Neville forgot about their anniversary.
---
Dialogue: John: Ave. Was there any homework for tomorrow?
Cassandra: hello :D Of course, as always :D
John: What exactly?
Cassandra: I'm not sure so I'll check it for you in 20minutes. 
John: Cool, thanks. Sorry I couldn't be there, but I was busy as fuck...my stupid boss as always was trying to piss me off
Cassandra: No problem, what did he do this time?
John: Nothing special, just the same as always, treating us like children, commanding to do this and that...
Cassandra: sorry to hear that. but why don't you just go to your chief and tell him everything?
John: I would, but I don't have any support from others, they are like goddamn pupets and pretend that everything's fine...I'm not gonna fix everything for everyone
Cassandra: I understand...Nevertheless, just try to ignore him. I know it might sound ridiculous as fuck, but sometimes there's nothing more you can do.
John: yeah I know...maybe some beer this week?
Cassandra: Sure, but I got some time after classes only...this week is gonna be busy
John: no problem, I can drive you home and we can go to some bar or whatever.
Cassandra: cool. ok, I got this homework. it's page 15 ex. 2 and 3, I also asked the others to study another chapter, especially the vocabulary from the very first pages. Just read it.
John: gosh...I don't know if I'm smart enough to do it :'D
Cassandra: you are, don't worry :P Just circle all the words you don't know and we'll continue on Monday.
John: ok...then I'll try my best :D
Cassandra: sure, if you will have any questions just either text or call me and I'll help you.
John: I hope I won't have to waste your time xD
Cassandra: you're not wasting my time, I'm your teacher, I'm here to help. This is what I get money for, also :P
John: just kidding :D ok, so i guess we'll stay in touch then
Cassandra: sure, have a nice evening :D
John: you too, se ya
Cassandra: Byeeeee
Summary: John didn't show up for class due to some work issues with his boss. Cassandra, his teacher told him which exercises to do, and which chapter to study. They are going to meet up for a beer sometime this week after class. 
---
Dialogue: Sarah: I found a song on youtube and I think you'll like it
James: What song?
Sarah: <file_other>
James: Oh. I know it! 
James: I heard it before in some compilation
Sarah: I can't stop playing it over and over
James: That's exactly how I know lyrics to all of the songs on my playlist :D
Sarah: Haha. No lyrics here though. Instrumental ;D
James: Instrumental songs are different kind of music. 
James: But you have to remember that the activity you do when you listen to this song
James: Is the actvity your brain will connect to the song
James: And everytime you play this song at home
James: You'll be thinking of your work
Sarah: Yeah, I know that. That's why we sometimes say - I used to like that song, but now it just reminds me of bad memories
James: Yup. Everytime you change your partner, you have to get rid of your favorite music :D
Sarah: Hahaha. True, true.
Summary: Sarah sends James an instrumental song he might like. James knows the song. The brain connects the songs to the context they were played in and brings to mind the associated memories.
---
Dialogue: Noah: When and where are we meeting? :)
Madison: I thought you were busy...?
Noah: Yeah, I WAS. I quit my job. 
Madison: No way! :o :o :o Why? I thought you liked it...?
Noah: Well, I used to, until my boss turned into a complete cock... Long story.
Summary: Noah wants to meet, he quit his job, because his boss was a dick.
---
Dialogue: Matt: Do you want to go for date?
Agnes: Wow! You caught me out with this question Matt.
Matt: Why?
Agnes: I simply didn't expect this from you.
Matt: Well, expect the unexpected.
Agnes: Can I think about it?
Matt: What is there to think about?
Agnes: Well, I don't really know you.
Matt: This is the perfect time to get to know eachother
Agnes: Well that's true.
Matt: So let's go to the Georgian restaurant in Kazimierz.
Agnes: Now your convincing me.
Matt: Cool, saturday at 6pm?
Agnes: That's fine.
Matt: I can pick you up on the way to the restaurant.
Agnes: That's really kind of you.
Matt: No problem.
Agnes: See you on saturday.
Matt: Yes, looking forward to it.
Agnes: Me too.
Summary: Matt invites Agnes for a date to get to know each other better. They'll go to the Georgian restaurant in Kazimierz on Saturday at 6 pm, and he'll pick her up on the way to the place.
---
"""