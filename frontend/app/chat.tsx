import { useState, useRef } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  FlatList, StyleSheet, KeyboardAvoidingView, Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import GuideAvatar from '../components/GuideAvatar';
import SuggestedChips from '../components/SuggestedChips';
import { Colors, Fonts } from '../constants/theme';
import { useSession } from '../store/session';

interface Message { id: string; from: 'guide' | 'user'; text: string }

const SUGGESTIONS = ['How fast can she run?', 'What does she eat?', 'Is she a danger to us?'];

const INITIAL_MESSAGES: Message[] = [
  { id: '1', from: 'guide', text: 'Great eye spotting her! I still have your photo open — ask me anything about this leopard.' },
];

const STUB_REPLIES: Record<string, string> = {
  'How fast can she run?':  "Leopards can sprint up to 58 km/h, but only for short bursts. They rely on stealth far more than speed.",
  'What does she eat?':     "Impala are the staple, but she'll take anything from hares to young zebra. She hoists kills into trees to protect them.",
  'Is she a danger to us?': "In a vehicle you're safe — she sees you as one large creature. Never exit the car near a leopard.",
};

export default function ChatScreen() {
  const { species, guideName } = useSession();
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState('');
  const listRef = useRef<FlatList>(null);

  function send(text: string) {
    if (!text.trim()) return;
    const userMsg: Message = { id: Date.now().toString(), from: 'user', text };
    const reply = STUB_REPLIES[text] ?? `That's a fascinating question about ${species}. Let me think on that in the field.`;
    const guideMsg: Message = { id: (Date.now() + 1).toString(), from: 'guide', text: reply };
    setMessages(prev => [...prev, userMsg, guideMsg]);
    setInput('');
    setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100);
  }

  const renderItem = ({ item }: { item: Message }) => (
    item.from === 'guide' ? (
      <View style={styles.guideRow}>
        <GuideAvatar name={guideName} />
        <View style={styles.guideBubbleWrap}>
          <Text style={styles.guideName}>{guideName.toUpperCase()} · GUIDE</Text>
          <View style={styles.guideBubble}>
            <Text style={styles.guideBubbleText}>{item.text}</Text>
          </View>
        </View>
      </View>
    ) : (
      <View style={styles.userBubble}>
        <Text style={styles.userBubbleText}>{item.text}</Text>
      </View>
    )
  );

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      {/* Dark header */}
      <View style={styles.header}>
        <View style={styles.statusBar}>
          <Text style={styles.statusText}>7:45</Text>
          <Text style={styles.statusText}>5G · 85%</Text>
        </View>
        <View style={styles.headerRow}>
          <TouchableOpacity onPress={() => router.back()}>
            <Text style={styles.backGlyph}>‹</Text>
          </TouchableOpacity>
          <View style={styles.headerThumb}>
            <Text style={{ fontSize: 22 }}>🐆</Text>
          </View>
          <View style={styles.headerMeta}>
            <Text style={styles.headerSpecies}>{species}</Text>
          </View>
          <Text style={styles.liveTag}>LIVE</Text>
        </View>
      </View>

      {/* Messages */}
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <FlatList
          ref={listRef}
          data={messages}
          renderItem={renderItem}
          keyExtractor={m => m.id}
          contentContainerStyle={styles.messageList}
          ListFooterComponent={
            <View style={{ marginTop: 2 }}>
              <SuggestedChips chips={SUGGESTIONS} onPress={send} />
            </View>
          }
        />

        {/* Input bar */}
        <View style={styles.inputBar}>
          <TextInput
            style={styles.input}
            placeholder={`Ask about this ${species.split(' ').pop()}…`}
            placeholderTextColor="#9a8c74"
            value={input}
            onChangeText={setInput}
            onSubmitEditing={() => send(input)}
            returnKeyType="send"
          />
          <TouchableOpacity style={styles.micBtn} onPress={() => send(input)}>
            <View style={styles.micBars}>
              {[11, 17, 11].map((h, i) => (
                <View key={i} style={[styles.micBar, { height: h }]} />
              ))}
            </View>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.cream },
  header: { backgroundColor: Colors.dark, paddingBottom: 16 },
  statusBar: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 24, height: 44, alignItems: 'center' },
  statusText: { fontFamily: Fonts.mono, fontSize: 12, color: Colors.cream },
  headerRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 18 },
  backGlyph: { fontFamily: Fonts.mono, fontSize: 22, color: 'rgba(243,236,222,0.8)' },
  headerThumb: { width: 44, height: 44, borderRadius: 9, backgroundColor: '#3a2f1d', alignItems: 'center', justifyContent: 'center' },
  headerMeta: { flex: 1 },
  headerSpecies: { fontFamily: Fonts.display, fontSize: 19, color: Colors.cream },
  liveTag: { fontFamily: Fonts.mono, fontSize: 9, letterSpacing: 1.4, color: 'rgba(243,236,222,0.55)' },
  messageList: { padding: 20, paddingBottom: 12, gap: 16 },
  guideRow: { flexDirection: 'row', gap: 10, maxWidth: '90%' },
  guideBubbleWrap: { flex: 1 },
  guideName: { fontFamily: Fonts.mono, fontSize: 8.5, letterSpacing: 1.2, color: Colors.muted, marginBottom: 5 },
  guideBubble: { backgroundColor: '#fff', borderWidth: 1, borderColor: 'rgba(28,22,13,0.07)', borderRadius: 4, borderTopLeftRadius: 14, borderBottomRightRadius: 14, borderBottomLeftRadius: 14, padding: 14 },
  guideBubbleText: { fontFamily: Fonts.body, fontSize: 16, lineHeight: 24, color: '#221a0f' },
  userBubble: { alignSelf: 'flex-end', maxWidth: '82%', backgroundColor: '#5a4326', borderRadius: 14, borderTopRightRadius: 4, padding: 14 },
  userBubbleText: { fontFamily: Fonts.body, fontSize: 16, lineHeight: 24, color: Colors.cream },
  inputBar: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 12, paddingHorizontal: 16, borderTopWidth: 1, borderTopColor: 'rgba(28,22,13,0.1)', backgroundColor: Colors.cream },
  input: { flex: 1, backgroundColor: '#fff', borderWidth: 1, borderColor: 'rgba(28,22,13,0.14)', borderRadius: 24, paddingHorizontal: 16, paddingVertical: 12, fontFamily: Fonts.body, fontSize: 15, color: '#1c160d' },
  micBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: Colors.dark, alignItems: 'center', justifyContent: 'center' },
  micBars: { flexDirection: 'row', alignItems: 'center', gap: 2.5 },
  micBar: { width: 3, backgroundColor: Colors.amber, borderRadius: 2 },
});
