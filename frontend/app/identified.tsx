import { ScrollView, View, Text, Image, TouchableOpacity, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import AudioPlayer from '../components/AudioPlayer';
import { Colors, Fonts } from '../constants/theme';
import { useSession } from '../store/session';

export default function IdentifiedScreen() {
  const { species, taxon, traits, confidence, guideName } = useSession();

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      {/* Hero image */}
      <View style={styles.hero}>
        <View style={styles.heroPlaceholder}>
          <Text style={styles.heroPlaceholderText}>📷</Text>
        </View>
        <LinearGradient
          colors={['rgba(15,11,5,0.5)', 'rgba(15,11,5,0)', 'rgba(15,11,5,0.1)', 'rgba(15,11,5,0.7)']}
          locations={[0, 0.3, 0.6, 1]}
          style={StyleSheet.absoluteFill}
        />
        {/* Status */}
        <View style={styles.statusBar}>
          <Text style={styles.statusText}>7:43</Text>
          <Text style={styles.statusText}>5G · 86%</Text>
        </View>
        {/* Nav */}
        <View style={styles.heroNav}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
            <Text style={styles.backGlyph}>‹</Text>
          </TouchableOpacity>
          <View style={styles.heroActions}>
            <Text style={styles.heroAction}>SAVE</Text>
            <Text style={styles.heroAction}>SHARE</Text>
          </View>
        </View>
        {/* Badge */}
        <View style={styles.badge}>
          <Text style={styles.badgeText}>{Math.round(confidence * 100)}% MATCH</Text>
        </View>
      </View>

      {/* Body */}
      <ScrollView style={styles.body} contentContainerStyle={styles.bodyContent} showsVerticalScrollIndicator={false}>
        <Text style={styles.taxon}>{taxon}</Text>
        <Text style={styles.speciesName}>{species}</Text>
        <Text style={styles.subtitle}>Solitary · nocturnal · ambush predator</Text>

        {/* Trait chips */}
        <View style={styles.traits}>
          {traits.map(t => (
            <View key={t} style={styles.traitChip}>
              <Text style={styles.traitText}>{t}</Text>
            </View>
          ))}
        </View>

        {/* Audio player */}
        <View style={styles.playerWrap}>
          <AudioPlayer guideName={guideName} />
        </View>

        {/* Drop-cap paragraph */}
        <Text style={styles.body1}>
          <Text style={styles.dropCap}>S</Text>
          he's a real beauty. That rosette-patterned coat is flawless camouflage in the dappled acacia light. Leopards are the great solitaries of the Mara — fiercely territorial, and strong enough to haul a kill twice their own weight straight up a tree, safe from lions and hyenas below.
        </Text>
      </ScrollView>

      {/* CTA */}
      <View style={styles.cta}>
        <TouchableOpacity style={styles.ctaBtn} onPress={() => router.push('/chat')}>
          <Text style={styles.ctaBtnText}>Ask {guideName} a question  →</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.cream },
  hero: { height: 328, position: 'relative' },
  heroPlaceholder: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: '#3a2f1d', alignItems: 'center', justifyContent: 'center' },
  heroPlaceholderText: { fontSize: 48 },
  statusBar: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 24, paddingTop: 4, height: 44, alignItems: 'center' },
  statusText: { fontFamily: Fonts.mono, fontSize: 12, color: Colors.cream },
  heroNav:   { position: 'absolute', top: 50, left: 18, right: 18, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  backBtn:   { width: 34, height: 34, borderRadius: 17, borderWidth: 1, borderColor: 'rgba(243,236,222,0.5)', alignItems: 'center', justifyContent: 'center' },
  backGlyph: { fontFamily: Fonts.mono, fontSize: 18, color: Colors.cream },
  heroActions: { flexDirection: 'row', gap: 16 },
  heroAction:  { fontFamily: Fonts.mono, fontSize: 11, letterSpacing: 1, color: 'rgba(243,236,222,0.85)' },
  badge:     { position: 'absolute', bottom: 18, left: 20 },
  badgeText: { fontFamily: Fonts.mono, fontWeight: '700', fontSize: 11, letterSpacing: 0.6, color: Colors.dark, backgroundColor: Colors.amber, paddingHorizontal: 11, paddingVertical: 6, borderRadius: 2 },
  body: { flex: 1 },
  bodyContent: { padding: 24, paddingBottom: 12, gap: 0 },
  taxon:     { fontFamily: Fonts.mono, fontSize: 10, letterSpacing: 1.6, color: Colors.muted, marginBottom: 4 },
  speciesName: { fontFamily: Fonts.display, fontWeight: '600', fontSize: 46, lineHeight: 46, color: Colors.text, marginBottom: 5 },
  subtitle:  { fontFamily: Fonts.bodyItalic, fontSize: 16, color: '#5a4d38', marginBottom: 16, fontStyle: 'italic' },
  traits:    { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 6 },
  traitChip: { borderWidth: 1, borderColor: 'rgba(28,22,13,0.2)', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 2 },
  traitText: { fontFamily: Fonts.mono, fontSize: 10, letterSpacing: 0.4, color: '#3a2f1d' },
  playerWrap:{ marginVertical: 18 },
  body1:     { fontFamily: Fonts.body, fontSize: 16.5, lineHeight: 26.7, color: '#2a2114' },
  dropCap:   { fontFamily: Fonts.display, fontSize: 58, lineHeight: 42, color: Colors.amber, fontWeight: '600' },
  cta:       { paddingHorizontal: 18, paddingVertical: 12, borderTopWidth: 1, borderTopColor: 'rgba(28,22,13,0.1)', backgroundColor: Colors.cream },
  ctaBtn:    { backgroundColor: Colors.dark, borderRadius: 30, paddingVertical: 15, alignItems: 'center', justifyContent: 'center' },
  ctaBtnText:{ fontFamily: Fonts.display, fontSize: 18, color: Colors.cream },
});
